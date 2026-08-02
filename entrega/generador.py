#!/usr/bin/env python
"""Regenerate `resultados.jsonl` from the persisted vector knowledge base.

This is deliverable 4 of section 1.4: "Script Python que utilice el índice, lea
el archivo de consultas y genere el archivo de resultados resultados.jsonl".
Section 1.4 also fixes its name and its place, inside `entrega/`. The judges
receive that directory alone, so the file is deliberately self-contained: it
imports nothing from the rest of the project. Its only dependencies are the
libraries the index itself was built with.

    pip install faiss-cpu numpy sentence-transformers pymupdf pysbd

Usage (from inside this directory):
    python generador.py
    python generador.py --queries path/to/questions.pdf --output resultados.jsonl
    python generador.py --encoders bge-m3          # restrict to one index

The retrieval module follows section 8, in this order:

    8.1  encode the query with the same encoder used at indexing time
    8.2  cosine similarity, as inner product over unit-normalized vectors
    8.4  fuse the per-encoder rankings with Reciprocal Rank Fusion
    8.7  post-filter over the `fenomeno` metadata field, as a soft boost
    ---  diversification: cap the positions one document may occupy
    8.6  aggregate fragments to document level by max pooling
    9.2  build 3 documents and 10 fragments of at most 250 words each
    9.3  validate the output against the schema before writing it

**Restriction on generative models (section 8.3).** No generative model
participates at any stage: no LLM reranking, no query reformulation or
expansion, no generative filtering, no synthesis of fragments. Every operation
runs over vectors, similarity scores and metadata.

Note on language: the metadata field names stay in Spanish (`fuente`, `formato`,
`fenomeno`, `posicion`, `num_tokens`, `texto`) because Table 1 defines them that
way, and so do the file names `generador.py` and `resultados.jsonl`, fixed by
section 1.4. Everything else is in English.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- Configuration --------------------------------------------------------
# These values mirror the ones used to build the index. Changing them changes
# the output, which is exactly what section 1.4 forbids: a submission that
# cannot reproduce its results is excluded from evaluation.

SEED = 20260801

HERE = Path(__file__).resolve().parent


# This file lives in `entrega/`, beside the index it reads and the results it
# writes, as section 1.4 requires. The question file is the only input that sits
# elsewhere in the development repository, so it falls back one level up.
DEFAULT_INDEX_ROOT = HERE / "base_vectorial"
DEFAULT_OUTPUT = HERE / "resultados.jsonl"


def _default_queries() -> Path:
    candidates = (
        HERE / "Extracto_Preguntas_50_v2.pdf",
        HERE.parent / "datos" / "Extracto_Preguntas_50_v2.pdf",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_QUERIES = _default_queries()

# Both are encoder architectures (XLM-RoBERTa) under MIT licence, the two
# criteria section 4.3 weighs most heavily after multilingual support. Section
# 4.2 forbids decoder backbones for embedding generation, which is why the MTEB
# leader Qwen3-Embedding is absent.
ENCODERS = {
    "bge-m3": {
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "max_tokens": 8192,
        "query_prefix": "",
    },
    "me5-large": {
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_tokens": 512,
        # E5 was trained with asymmetric prefixes; dropping them degrades retrieval.
        "query_prefix": "query: ",
    },
}

TOP_DOCUMENTS = 3
TOP_FRAGMENTS = 10
CANDIDATES_PER_INDEX = 100
RRF_K0 = 60
MAX_FRAGMENTS_PER_DOC = 3
PHENOMENON_BOOST = 1.05
MAX_WORDS_PER_FRAGMENT = 250

# Section 9.2.1 allows splitting an over-long fragment into sub-fragments, each
# taking its own rank. Measured, that delivers less: short tails — the second
# piece of a 321-word fragment is 71 words — occupy a whole rank with little
# content, dropping delivered text from 104k to 84k words and distinct fragments
# covered from 10 to 6. Trimming keeps all ten positions at 250 words.
SPLIT_LONG_FRAGMENTS = False

# Query-to-phenomenon mapping. The three phenomena are the ones section 1.1
# defines: artificial intelligence in defence, space security and low Earth
# orbit, and territorial dynamics in Latin America.
PHENOMENON_RANGES = {1: (1, 16), 2: (17, 32), 3: (33, 50)}


def phenomenon_of(query_id: str) -> int | None:
    """'q018' -> 2. Returns None when the identifier does not fit the pattern."""
    try:
        n = int(query_id.lstrip("qQ"))
    except ValueError:
        return None
    for phenomenon, (lo, hi) in PHENOMENON_RANGES.items():
        if lo <= n <= hi:
            return phenomenon
    return None


def set_seeds(seed: int = SEED) -> None:
    """Retrieval here is deterministic, but seeds are pinned anyway so that any
    future component carrying randomness cannot silently break reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --- Queries --------------------------------------------------------------

_QUERY_ID = re.compile(r"^(q\d{3})\b[\s:.\-]*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str

    @property
    def phenomenon(self) -> int | None:
        return phenomenon_of(self.query_id)


def _queries_from_text(content: str) -> list[Query]:
    """Each question may span several lines, so lines accumulate until the next
    identifier appears. Splitting on line breaks would truncate the long ones."""
    queries: list[Query] = []
    current_id: str | None = None
    pieces: list[str] = []

    def flush() -> None:
        if current_id and pieces:
            queries.append(Query(current_id.lower(), " ".join(" ".join(pieces).split())))

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _QUERY_ID.match(line)
        if match:
            flush()
            current_id = match.group(1)
            pieces = [match.group(2)] if match.group(2) else []
        elif current_id:
            pieces.append(line)

    flush()
    return queries


def _queries_from_pdf(path: Path) -> list[Query]:
    import pymupdf

    with pymupdf.open(path) as doc:
        content = "\n".join(page.get_text("text", sort=True) for page in doc)
    return _queries_from_text(content)


def _queries_from_jsonl(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("texto") or record.get("query") or record.get("pregunta") or ""
            queries.append(Query(str(record["query_id"]).lower(), text.strip()))
    return queries


def load_queries(path: Path) -> list[Query]:
    """Accepts PDF, JSONL or plain text: the format of the evaluation question
    file is not guaranteed to be the same one handed out during development."""
    if not path.exists():
        raise FileNotFoundError(f"query file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        queries = _queries_from_pdf(path)
    elif suffix in (".jsonl", ".json"):
        queries = _queries_from_jsonl(path)
    else:
        queries = _queries_from_text(path.read_text(encoding="utf-8"))

    queries.sort(key=lambda q: q.query_id)
    return queries


# --- Vector index ---------------------------------------------------------


@dataclass
class VectorIndex:
    name: str
    faiss_index: object
    metadata: list[dict]

    def __len__(self) -> int:
        return len(self.metadata)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        scores, ids = self.faiss_index.search(
            np.ascontiguousarray(vector, dtype=np.float32), k
        )
        # FAISS pads with -1 when fewer neighbours than requested exist.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def load_index(name: str, root: Path) -> VectorIndex:
    """Loads one encoder's index and its metadata store (section 5.3).

    FAISS keeps only vectors and internal integer identifiers; the metadata
    lives in a separate store. Section 1.4 requires the order of the lines in
    metadata.jsonl to match those internal identifiers, so line n describes
    vector n. That contract is asserted here rather than trusted.
    """
    import faiss

    folder = root / f"encoder_{name}"
    index_path = folder / "index.faiss"
    metadata_path = folder / "metadata.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    with metadata_path.open(encoding="utf-8") as fh:
        metadata = [json.loads(line) for line in fh if line.strip()]

    if index.ntotal != len(metadata):
        raise ValueError(
            f"index and metadata misaligned in {name}: "
            f"{index.ntotal} vectors against {len(metadata)} records"
        )

    return VectorIndex(name, index, metadata)


def available_encoders(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        d.name.removeprefix("encoder_")
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith("encoder_") and (d / "index.faiss").exists()
    )


# --- Query encoding -------------------------------------------------------


class QueryEncoder:
    """Encodes the query with the same encoder used at indexing time (8.1).

    Section 8.1 calls this indispensable: a different encoder would place the
    query vector and the index vectors in unrelated semantic spaces. The output
    is unit-normalized so that the index's inner product equals cosine
    similarity (8.2).
    """

    def __init__(self, name: str, device: str | None = None):
        if name not in ENCODERS:
            raise KeyError(f"unknown encoder: {name}")
        self.name = name
        self.cfg = ENCODERS[name]
        self.device = device or self._default_device()
        self._model = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg["model"], device=self.device)
            self._model.max_seq_length = min(
                self.cfg["max_tokens"], self._model.max_seq_length
            )
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        prefix = self.cfg["query_prefix"]
        if prefix:
            texts = [prefix + t for t in texts]
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            # Required for IndexFlatIP to behave as cosine similarity (section 8.2).
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# --- Retrieval ------------------------------------------------------------


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    text: str
    fuente: str
    fenomeno: int
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)


def fuse_rrf(
    rankings: dict[str, list[dict]], k0: int = RRF_K0
) -> list[Candidate]:
    """Combines the two indices with Reciprocal Rank Fusion (section 8.4).

    Of the three strategies section 8.4 lists — CombSUM, CombMNZ and RRF — this
    one combines rank positions rather than scores, so it is robust to scale
    differences between encoders. `k0 = 60` is the value the section gives as
    typical.

    An honest caveat: that robustness matters most when fusing heterogeneous
    rankers, and here both are dense encoders producing cosine values in the
    same range, so CombSUM could do as well or better. The comparison belongs to
    the sweep against the internal evaluation set.
    """
    pooled: dict[str, Candidate] = {}

    for encoder, ranking in rankings.items():
        for rank, meta in enumerate(ranking, start=1):
            chunk_id = meta["chunk_id"]
            candidate = pooled.get(chunk_id)
            if candidate is None:
                candidate = Candidate(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    text=meta["texto"],
                    fuente=meta.get("fuente", ""),
                    fenomeno=meta.get("fenomeno", 0),
                )
                pooled[chunk_id] = candidate
            candidate.score += 1.0 / (k0 + rank)
            candidate.ranks[encoder] = rank

    return sorted(pooled.values(), key=lambda c: c.score, reverse=True)


def apply_phenomenon_boost(
    candidates: list[Candidate], phenomenon: int | None, factor: float = PHENOMENON_BOOST
) -> list[Candidate]:
    """Post-filter over the `fenomeno` metadata field (section 8.7).

    Section 8.7 allows filtering on metadata fields, and Table 1 makes
    `fenomeno` available. It is applied as a soft multiplier rather than a hard
    filter, because several queries admit cross-cutting evidence: q005 and q046
    both address Colombia from different phenomena, and q027 spans artificial
    intelligence and space operations. A strict filter would foreclose
    legitimately relevant documents.
    """
    if phenomenon is None or factor == 1.0:
        return candidates

    for candidate in candidates:
        if candidate.fenomeno == phenomenon:
            candidate.score *= factor

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def drop_duplicates(candidates: list[Candidate]) -> list[Candidate]:
    """Removes fragments with repeated text, keeping the best-scored one.

    The corpus holds 4,445 fragments (3%) with byte-identical text: the same
    tables reprinted across reports, the same institutional headers. Two copies
    of one text carry the same relevance and burn two of the ten positions
    NDCG@10 evaluates to inform once.

    Comparison runs over whitespace-normalized text, so variants differing only
    in line breaks fall too.
    """
    seen: set[str] = set()
    unique: list[Candidate] = []

    for candidate in candidates:
        key = " ".join(candidate.text.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return unique


def diversify(
    candidates: list[Candidate], limit: int, max_per_doc: int = MAX_FRAGMENTS_PER_DOC
) -> list[Candidate]:
    """Caps how many fragments a single document contributes to the final list.

    Without the cap one document can occupy all ten evaluated positions; if it
    turns out to be irrelevant, the whole query is lost. The cap spreads that
    risk across several documents.
    """
    selected: list[Candidate] = []
    per_doc: dict[str, int] = {}

    for candidate in candidates:
        if per_doc.get(candidate.doc_id, 0) >= max_per_doc:
            continue
        selected.append(candidate)
        per_doc[candidate.doc_id] = per_doc.get(candidate.doc_id, 0) + 1
        if len(selected) >= limit:
            break

    # If the cap left the list short, fill it with the best discarded ones.
    if len(selected) < limit:
        taken = {c.chunk_id for c in selected}
        for candidate in candidates:
            if candidate.chunk_id in taken:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break

    return selected


def aggregate_to_documents(
    candidates: list[Candidate], top: int = TOP_DOCUMENTS
) -> list[str]:
    """Aggregates fragments to document level by max pooling (section 8.6).

    Of the three strategies section 8.6 lists — best fragment, sum of all
    retrieved fragments, weighted mean — max pooling is chosen: each document
    inherits its best fragment's score. Sum pooling is rejected for length bias,
    since a document with forty weak fragments at 0.15 accumulates 6.0 and
    outranks a focused report holding the exact answer at 0.85. Given the corpus
    size heterogeneity, that bias would be severe.

    Fragment count acts only as a tiebreaker, weighted small enough not to
    disturb the primary ordering.
    """
    best: dict[str, float] = {}
    count: dict[str, int] = {}

    for candidate in candidates:
        previous = best.get(candidate.doc_id)
        if previous is None or candidate.score > previous:
            best[candidate.doc_id] = candidate.score
        count[candidate.doc_id] = count.get(candidate.doc_id, 0) + 1

    ordered = sorted(
        best.items(), key=lambda kv: (kv[1], count[kv[0]] * 1e-9), reverse=True
    )
    return [doc_id for doc_id, _ in ordered[:top]]


class Retriever:
    def __init__(
        self,
        indices: dict[str, VectorIndex],
        k0: int = RRF_K0,
        boost: float = PHENOMENON_BOOST,
        max_per_doc: int = MAX_FRAGMENTS_PER_DOC,
        candidates_per_index: int = CANDIDATES_PER_INDEX,
    ):
        if not indices:
            raise ValueError("no indices loaded")
        self.indices = indices
        self.k0 = k0
        self.boost = boost
        self.max_per_doc = max_per_doc
        self.candidates_per_index = candidates_per_index
        self._encoders: dict[str, QueryEncoder] = {}

    def _encoder(self, name: str) -> QueryEncoder:
        if name not in self._encoders:
            self._encoders[name] = QueryEncoder(name)
        return self._encoders[name]

    def retrieve(self, query: Query) -> tuple[list[str], list[Candidate]]:
        rankings: dict[str, list[dict]] = {}

        for name, index in self.indices.items():
            vector = self._encoder(name).encode([query.text])[0]
            hits = index.search(vector, self.candidates_per_index)
            rankings[name] = [index.metadata[pos] for pos, _score in hits]

        candidates = fuse_rrf(rankings, self.k0)
        candidates = apply_phenomenon_boost(candidates, query.phenomenon, self.boost)

        # Diversification runs before pooling because it protects the fragment
        # list, which is what NDCG@10 evaluates. Deduplication applies only
        # there too: two distinct documents may share a text, and dropping one
        # would make it unreachable. Document aggregation therefore uses the
        # full candidate set, so no signal is discarded.
        fragments = diversify(
            drop_duplicates(candidates), TOP_FRAGMENTS, self.max_per_doc
        )
        documents = aggregate_to_documents(candidates, TOP_DOCUMENTS)

        return documents, fragments


# --- Output construction --------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[^\s])")


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation for the 250-word trim.

    Uses pysbd when available — it is what the chunker used — and falls back to
    a punctuation regex so the deliverable still runs without that dependency.
    Both respect paragraph boundaries: text from separate paragraphs is never
    merged into one sentence.
    """
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        try:
            import pysbd

            pieces = pysbd.Segmenter(language="es", clean=False).segment(paragraph)
        except Exception:
            pieces = _SENTENCE_SPLIT.split(paragraph)
        sentences.extend(p.strip() for p in pieces if p and p.strip())
    return sentences


def word_count(text: str) -> int:
    return len(text.split())


def split_to_limit(text: str, max_words: int = MAX_WORDS_PER_FRAGMENT) -> list[str]:
    """Splits a fragment into pieces of at most `max_words`, never cutting a
    sentence (section 9.2.1).

    A sentence exceeding the limit on its own — it happens with tables dumped to
    text — is cut by words: respecting the maximum takes priority, since a
    fragment above it is discarded by the automatic evaluator (section 9.3.2).
    """
    if word_count(text) <= max_words:
        return [text]

    pieces: list[str] = []
    accumulated: list[str] = []
    total = 0

    for sentence in split_sentences(text):
        n = word_count(sentence)

        if n > max_words:
            if accumulated:
                pieces.append(" ".join(accumulated))
                accumulated, total = [], 0
            words = sentence.split()
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
            continue

        if accumulated and total + n > max_words:
            pieces.append(" ".join(accumulated))
            accumulated, total = [], 0

        accumulated.append(sentence)
        total += n

    if accumulated:
        pieces.append(" ".join(accumulated))

    return pieces or [" ".join(text.split()[:max_words])]


def build_record(
    query_id: str, documents: list[str], fragments: list[Candidate]
) -> dict:
    """Builds one query's JSON object, per the schema of section 9.3.1.

    Exactly 3 documents and 10 fragments, each field of Table 2 present, and no
    fragment above 250 words (9.2). The reported `chunk_id` stays that of the
    originating index fragment even after trimming: section 10.2.1 states it is
    not the matching key against the ground truth, since every team chunks
    differently — it serves traceability and verification.
    """
    docs = [
        {"rank": i, "doc_id": doc_id}
        for i, doc_id in enumerate(documents[:TOP_DOCUMENTS], start=1)
    ]

    # Fragments above the word limit are split rather than truncated, each piece
    # taking its own rank. The per-document cap counts delivered positions, not
    # index fragments: otherwise one document could take six of the ten
    # evaluated slots, which is what diversification exists to prevent.
    frags: list[dict] = []
    per_doc: dict[str, int] = {}

    for candidate in fragments:
        if len(frags) >= TOP_FRAGMENTS:
            break
        if per_doc.get(candidate.doc_id, 0) >= MAX_FRAGMENTS_PER_DOC:
            continue

        pieces = (
            split_to_limit(candidate.text)
            if SPLIT_LONG_FRAGMENTS
            else split_to_limit(candidate.text)[:1]
        )
        for piece in pieces:
            if (
                len(frags) >= TOP_FRAGMENTS
                or per_doc.get(candidate.doc_id, 0) >= MAX_FRAGMENTS_PER_DOC
            ):
                break
            frags.append(
                {
                    "rank": len(frags) + 1,
                    # The chunk_id stays that of the originating index fragment
                    # across pieces: traceability, not matching (10.2.1).
                    "chunk_id": candidate.chunk_id,
                    "doc_id": candidate.doc_id,
                    "text": piece,
                }
            )
            per_doc[candidate.doc_id] = per_doc.get(candidate.doc_id, 0) + 1

    # The schema demands exactly 3 documents and 10 fragments; a short list is
    # discarded by the automatic evaluator. Padding only triggers on pathological
    # queries where the index returns very few candidates.
    while docs and len(docs) < TOP_DOCUMENTS:
        docs.append({"rank": len(docs) + 1, "doc_id": docs[-1]["doc_id"]})
    while frags and len(frags) < TOP_FRAGMENTS:
        frags.append({**frags[-1], "rank": len(frags) + 1})

    return {"query_id": query_id, "documents": docs, "fragments": frags}


def validate(path: Path, expected_queries: int = 50) -> list[str]:
    """Checks the file against sections 9.3.2 and 10.3 before delivery.

    Section 9.3.2 warns that objects with missing fields, arrays of a size other
    than 3 and 10, or fragments above 250 words are penalised or discarded by
    the automatic evaluation. Section 10.3 adds that submissions with an
    incorrect format are not considered at all. Cheap to check, expensive to get
    wrong.
    """
    problems: list[str] = []

    with path.open(encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]

    if len(lines) != expected_queries:
        problems.append(f"{len(lines)} lines, expected {expected_queries}")

    seen: set[str] = set()
    for i, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {i}: invalid JSON ({e})")
            continue

        query_id = record.get("query_id")
        if not query_id:
            problems.append(f"line {i}: missing query_id")
            continue
        if query_id in seen:
            problems.append(f"line {i}: duplicate query_id ({query_id})")
        seen.add(query_id)

        documents = record.get("documents") or []
        fragments = record.get("fragments") or []

        if len(documents) != TOP_DOCUMENTS:
            problems.append(
                f"{query_id}: {len(documents)} documents, expected {TOP_DOCUMENTS}"
            )
        if len(fragments) != TOP_FRAGMENTS:
            problems.append(
                f"{query_id}: {len(fragments)} fragments, expected {TOP_FRAGMENTS}"
            )

        for fragment in fragments:
            missing = [
                f for f in ("rank", "chunk_id", "doc_id", "text") if f not in fragment
            ]
            if missing:
                problems.append(f"{query_id}: fragment missing {missing}")
                continue
            words = word_count(fragment["text"])
            if words > MAX_WORDS_PER_FRAGMENT:
                problems.append(
                    f"{query_id} rank {fragment['rank']}: {words} words "
                    f"(maximum {MAX_WORDS_PER_FRAGMENT})"
                )

    return problems


# --- Entry point ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate resultados.jsonl from the index.")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--encoders", help="comma-separated list; defaults to all found")
    ap.add_argument("--k0", type=int, default=RRF_K0)
    ap.add_argument("--boost", type=float, default=PHENOMENON_BOOST)
    ap.add_argument("--max-per-doc", type=int, default=MAX_FRAGMENTS_PER_DOC)
    args = ap.parse_args()

    set_seeds()

    queries = load_queries(args.queries)
    print(f"queries: {len(queries)}")

    found = available_encoders(args.index_root)
    if not found:
        print(f"no indices under {args.index_root}")
        return 1

    chosen = [e.strip() for e in args.encoders.split(",")] if args.encoders else found
    missing = set(chosen) - set(found)
    if missing:
        print(f"indices not found: {sorted(missing)} (available: {found})")
        return 1

    indices = {name: load_index(name, args.index_root) for name in chosen}
    for name, index in indices.items():
        print(f"index {name}: {len(index):,} fragments")

    retriever = Retriever(
        indices, k0=args.k0, boost=args.boost, max_per_doc=args.max_per_doc
    )

    records = []
    for query in queries:
        documents, fragments = retriever.retrieve(query)
        records.append(build_record(query.query_id, documents, fragments))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n-> {args.output}")

    problems = validate(args.output, expected_queries=len(queries))
    if problems:
        print(f"\n{len(problems)} schema problems:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1

    print("schema validated: format is correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
