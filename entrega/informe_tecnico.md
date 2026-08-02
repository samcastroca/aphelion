# Vector Knowledge Base — Technical Report

**CODEFEST AD ASTRA 2026 · Stage 1**

---

## 1. Corpus characterization

All design decisions below follow from measurements taken over the full ADL
inventory (1826 files, verified 100% against disk).

| Format | Files | Extractable text |
|---|---:|---:|
| PDF | 760 | 87.5M characters |
| JSON | 964 | 9.5M characters |
| CSV | 26 | 150M characters |
| PBF | 73 | map feature attributes |
| XLSX | 6 | AI Index datasets |
| Image | 9 | OCR |
| **Total** | **1826** | **~257M characters (~64M tokens)** |

Four measured facts shaped the architecture:

**Language asymmetry is the central problem.** All 50 evaluation queries are in
Spanish, while 1019 documents are in English, 541 in Spanish and 99 in Portuguese.
Cross-lingual retrieval is therefore not an edge case but the dominant
requirement, and it drove encoder selection above every other criterion.

**53 PDFs carry no text layer.** 48 of them are scanned Defensoría del Pueblo
early-warning reports — precisely the material answering queries q033–q050.
Skipping them would blind the system across the entire third phenomenon.

**Extreme size heterogeneity.** JSON alerts of ~1.3k characters coexist with
RESDAL atlases of several hundred pages. This ruled out sum pooling for
document-level aggregation and required chunking that tolerates both extremes.

**Five CSV files dominate the raw index.** The PubMed bibliographic exports
account for 83,350 of 149,571 fragments (56%), all of them biomedical reference
listings unrelated to any evaluation query. They are retained in the index for
corpus fidelity, but flagged via the `formato` metadata field so they can be
excluded by post-filter once ground-truth measurement is available.

## 2. Document identification

Identifiers are not invented. The scheme ADL defines in `Indice_Datos_Codefest.xlsx`
is replicated exactly:

```
doc_id = {phenomenon}-{observatory_code}-{sequence:03d}   e.g. F1-CSET-001
fuente = the inventory's "Nombre estandarizado" field, verbatim
```

This matters because section 10.2.1 establishes that ground-truth matching occurs
through `fuente`, not through `doc_id`. Any divergence from ADL's original
filename invalidates the F1@3 metric regardless of retrieval quality.

Ingestion iterates over the inventory rather than over a directory listing. This
excludes by construction the artifacts that would otherwise contaminate the index:
`.DS_Store` files, the inventory spreadsheets, and — critically — the evaluation
questions PDF, whose indexing would place the exam inside the corpus.

A verified caveat: 59 standardized names are duplicated across different folders
(186 files), holding genuinely different content — confirmed by MD5 hashing, which
also established that the corpus contains no exact duplicates. The literal name is
reported in `fuente`; the full path is preserved in a separate field for internal
traceability.

## 3. Text extraction

| Format | Approach |
|---|---|
| PDF | PyMuPDF with reading-order block sorting, which matters for the two-column layouts of RESDAL atlases and CSIS reports |
| JSON (article) | `title` + `body_paragraphs`/`body_text` to the body; `url`, `date`, `authors`, `tags` retained as metadata rather than mixed into the text |
| JSON (catalog) | Each record in the list emitted as an independent block |
| CSV / XLSX | Each row as a unit, with `column: value` pairs so every value keeps its header as context |
| PBF | Feature attributes as `key: value`; duplicates dropped within each tile |
| Image | OCR |

Defensoría alerts required special handling: their substantive content — risk
theme, municipalities, armed groups — lives in the `alerta_meta` object rather
than in `body_paragraphs`. Extracting only the body would have reduced each alert
to a stray paragraph.

PBF tiles are kept as separate documents despite repeating across zoom levels,
because ADL assigns a distinct `doc_id` to each tile. Deduplication is applied to
features *within* a tile, not across tiles: removing tiles would make documents
unreachable that the ground truth may mark relevant.

### Cleaning

Boilerplate is removed by **line frequency** rather than fixed patterns. A short
line repeating four or more times within a document is an institutional header or
a chart axis label; a body sentence is not. The criterion is structural and works
identically across all three corpus languages. On a representative AI Index
chapter this removed 14% of lines — headers, figure sources and axis labels —
while leaving prose intact.

Dominant language is estimated from function words, with Spanish/Portuguese
disambiguated through exclusive markers.

## 4. OCR

Two backends were implemented to allow an evidence-based choice:
`baidu/Unlimited-OCR` (3B parameters, MIT license, layout-aware, R-SWA attention
allowing constant-memory processing of long documents) and Tesseract with Spanish
language data.

The dominant selection criterion is **absence of hallucination**, not average
accuracy. A vision-language OCR model that invents text injects false evidence
into the index, which could then be presented to the evaluators as retrieved
support. Tesseract, when it fails, produces obvious noise instead. The comparison
harness reports diacritic density (a model losing Spanish accents is failing even
when output looks fluent), repeated n-gram ratio (the characteristic VLM
generation-loop failure mode) and non-printable character ratio.

**Regulatory framing.** OCR belongs to preprocessing (section 2.1), where the
specification explicitly recommends it. The prohibition on decoder architectures
governs embedding generation (section 4.2) and the retrieval module (section 8.3);
neither stage employs a generative model in this system. The use of a
vision-language model for OCR is declared here for transparency.

## 5. Chunking

**Configuration: 512 tokens, 15% overlap, sentence-boundary cuts.**

Section 3.3 forbids a sentence from crossing a fragment boundary. This rules out
cutting by token count and requires accumulating whole sentences: when the next
sentence does not fit the budget, the fragment closes where the previous one
ended. Overlap is implemented by carrying trailing sentences forward, so overlap
boundaries also land on sentence limits.

Token counting uses the encoder's own tokenizer rather than a word-based
approximation, since the budget that matters is the model's.

Two implementation details proved necessary at corpus scale:

- The corpus contains blocks exceeding 140,000 tokens with no internal punctuation
  (PBF map dumps, very wide CSV rows). Splitting these by incrementally appending
  words and recounting is quadratic; the implementation tokenizes once and cuts
  windows over the offset map. This reduced full-corpus chunking from 147 to 94
  minutes with byte-identical output.
- Summing per-sentence token counts is not equivalent to tokenizing the joined
  text — the tokenizer may merge or split pieces at junctions. Fragments near the
  budget are therefore recounted and split if they exceed it, which prevents
  silent truncation under short-window encoders such as mE5-large.

**Justification.** Comparative evaluations over technical PDFs place recursive
512-token chunking with sentence-aligned overlap as the best compromise across the
two evaluated granularities (fragment-level F1 0.92, document-level 0.86).
Semantic chunking is explicitly rejected: despite strong fragment-level
performance (0.91), it collapses at document level (0.42) by producing ~43-token
fragments that destroy context — precisely the F1@3 metric under evaluation.

**Parameter subject to empirical sweep.** Chunk size exhibits tension between the
two metrics: 512 tokens favors NDCG@10 while 1024 favors F1@3. Since the Borda
count weights both equally, the value is set empirically by sweeping {384, 512,
768} against the internal evaluation set.

Resulting corpus: **149,571 fragments, 72M tokens, median 499 tokens per
fragment.** Index coverage is 1758 of 1826 documents; the remainder are the 60
awaiting OCR and 8 JSON files containing only dates.

## 6. Semantic encoding

| Model | License | Architecture | Dim. | Context | Role |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | MIT | XLM-RoBERTa (encoder) | 1024 | 8192 | Primary |
| `intfloat/multilingual-e5-large` | MIT | XLM-RoBERTa (encoder) | 1024 | 512 | Complementary |

**BGE-M3 as primary:** it outperforms mE5-large on Spanish retrieval (0.727 versus
0.660 on MIRACL-VISION), its 8192-token window removes any constraint on chunking,
and it produces sparse alongside dense representations in a single forward pass,
providing lexical sensitivity without a separate index. That lexical component
matters here: the queries are dense with acronyms (NBQR, RPO, GEO, DIH, ASAT,
GAO/GAOR/GDO) and proper nouns (Chocó, Antioquia, Arauca, Norte de Santander).

**Second encoder rationale:** mE5-large maintains a vector space more strongly
partitioned by language, reducing the tendency to return documents in a language
other than the query's. It complements BGE-M3's known weakness along that
dimension. E5 requires asymmetric prefixes (`query: ` / `passage: `), applied
transparently at the encoder boundary.

**Models rejected:**

- **Qwen3-Embedding** leads multilingual MTEB (70.88 retrieval) but derives from an
  autoregressive decoder backbone. **Explicitly forbidden by section 4.2.**
- **Jina Embeddings v3** carries a CC-BY-NC-4.0 license, incompatible with the
  licensing criterion of section 4.3.
- **LaBSE** was trained to align parallel sentence pairs and lacks any notion of
  asymmetric topical relevance, scoring 18.80 on zero-shot retrieval.

## 7. Vector index

**`IndexFlatIP` over unit-normalized vectors**, one independent index per encoder.

At this corpus size exhaustive search is exact and resolves in milliseconds.
Approximate indices (IVF, HNSW) trade exactness for speed this volume does not
require, and here ranking exactness *is* the evaluated metric. Prior normalization
makes inner product equivalent to cosine similarity (section 8.2).

FAISS stores only vectors and integer identifiers. The metadata link is
positional: line *n* of `metadata.jsonl` describes vector *n*. Insertion order is
part of the contract, and index/metadata alignment is asserted on both save and
load.

## 8. Retrieval

```
per-index search → RRF fusion → phenomenon boost → diversification
                 → top-10 fragments → max pooling → top-3 documents
```

**Reciprocal Rank Fusion** combines the encoders. RRF operates on positions rather
than scores, making it immune to scale differences between distinct vector spaces:
BGE-M3 and E5 cosine values are not mutually comparable, but their orderings are.
`k₀ = 60` is the starting point, tuned against the internal evaluation set.

**Phenomenon boost, not hard filter.** The query→phenomenon mapping is known
(q001–q016 → F1, q017–q032 → F2, q033–q050 → F3), but several queries admit
cross-cutting evidence: q005 and q046 both address Colombia from different
phenomena, and q027 spans artificial intelligence and space operations. A strict
filter would foreclose legitimately relevant documents, so a soft multiplier is
applied instead.

**Diversification** caps how many fragments a single document contributes to the
top-10. Without it one document can occupy all ten evaluated positions; if that
document proves irrelevant, the entire query is lost.

**Max pooling** for document aggregation: each document inherits its best
fragment's score. Sum pooling is rejected for length bias — a document with forty
weak fragments (0.15 each) accumulates 6.0 and outranks a focused report holding
the exact answer at 0.85. Given the corpus size heterogeneity documented in
section 1, that bias would be severe. Fragment count acts only as a tiebreaker,
weighted small enough not to disturb the primary ordering.

**No generative model participates at any stage:** no LLM reranking, no query
expansion, no generative filtering, no synthesis. All operations run over vectors,
similarity scores and metadata, as section 8.3 requires.

### Output construction

Fragments exceeding 250 words are subdivided respecting sentence boundaries; the
reported `chunk_id` remains that of the originating index fragment, serving
traceability rather than matching (section 10.2.1). The generated file is
validated against the section 9.3 schema before delivery: exactly 50 lines, 3
documents and 10 fragments per query, no fragment above the word limit.

## 9. Internal evaluation

The official ground truth is not public. One is annotated over the 50 real
queries, which permits optimizing against the actual evaluation distribution
rather than synthetic proxies.

The annotation pool is built from the union of each encoder's top-N **separately**,
not from the fused system output. Annotating only the final output would bias
measurement toward what the current configuration already retrieves: any future
change surfacing new documents would appear as unannotated noise and be penalized
unfairly.

NDCG@10 and F1@3 are implemented as literal transcriptions of the section 10.2
formulas, including the `min(|Dq|, 3)` recall normalization — without which a
query holding five relevant documents would be capped at 0.6 even when the system
returns the three best possible.

## 10. Reproducibility

`generador.py` loads the persisted indices, reads the query file and regenerates
`resultados.jsonl` without reindexing. Random seeds are fixed and model versions
pinned. The specification excludes from evaluation any submission that cannot
reproduce its results; this is treated as an elimination criterion rather than a
recommendation.
