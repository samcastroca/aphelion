#!/usr/bin/env python
"""Genera resultados.jsonl a partir de la base vectorial ya construida.

Lee las 50 consultas, las busca contra los índices FAISS persistidos y escribe
el archivo de resultados sin volver a indexar el corpus.

El archivo es autónomo a propósito: no importa nada del resto del proyecto,
porque tiene que funcionar con solo esta carpeta y el índice al lado. Sus únicas
dependencias son las bibliotecas con las que se construyó la base.

    pip install faiss-cpu numpy sentence-transformers pymupdf pysbd

Uso (desde esta carpeta):
    python generador.py
    python generador.py --queries ruta/preguntas.pdf --output resultados.jsonl
    python generador.py --encoders bge-m3          # un solo índice

El flujo de recuperación es:

    codificar la consulta con el mismo encoder de la indexación
    buscar en cada índice y fusionar los rankings con RRF
    realzar el fenómeno esperado y diversificar por documento
    top-10 de fragmentos; top2 pooling para el top-3 de documentos
    validar el formato antes de escribir

No interviene ningún modelo generativo en ninguna etapa: no hay reordenamiento
por LLM, reformulación de la consulta, filtrado generativo ni síntesis. Todo se
resuelve sobre vectores, similitudes y metadata.

Los nombres de los campos de metadata (`fuente`, `formato`, `fenomeno`,
`posicion`, `num_tokens`, `texto`) van en español porque así los define el reto.
El resto del código está en inglés.
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

# --- Configuración ---------------------------------------------------------
# Estos valores son los mismos con los que se construyó el índice. Cambiar
# cualquiera cambia la salida, y entonces los resultados dejan de reproducirse.

SEED = 20260801

HERE = Path(__file__).resolve().parent

# El índice y el archivo de resultados viven junto a este script. El archivo de
# preguntas es el único insumo que en el repositorio de desarrollo está en otra
# carpeta, así que se busca también un nivel más arriba.
DEFAULT_INDEX_ROOT = HERE / "base_vectorial"
DEFAULT_OUTPUT = HERE / "resultados.jsonl"


def _default_queries() -> Path:
    candidates = (
        HERE / "Extracto_Preguntas_50_v2.pdf",
        HERE.parent / "data" / "Extracto_Preguntas_50_v2.pdf",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_QUERIES = _default_queries()

# Los dos son arquitecturas encoder (XLM-RoBERTa) con licencia MIT. Quedan fuera
# los modelos derivados de un decoder, por más que lideren MTEB.
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
        # E5 se entrenó con prefijos asimétricos; omitirlos degrada bastante.
        "query_prefix": "query: ",
    },
}

TOP_DOCUMENTS = 3
TOP_FRAGMENTS = 10
# Un documento que no entre en este pool no lo recupera ninguna etapa posterior.
# Importa sobre todo para el top-3 de documentos: uno que quede en el puesto 95
# de un índice y en el 120 del otro solo suma consenso si los dos lo traen.
# Ampliarlo no desplaza el top-10 —con k0=60 el puesto 150 aporta 1/210 frente a
# 1/65 del puesto 5— y la búsqueda es exhaustiva de todos modos.
CANDIDATES_PER_INDEX = 200
RRF_K0 = 60
MAX_FRAGMENTS_PER_DOC = 3
PHENOMENON_BOOST = 1.05

# Cómo se colapsan los fragmentos de un documento en una puntuación: `max` toma
# el mejor, `topN` promedia los N mejores.
#
# `top2` es lo que se entrega. Medido sobre el corpus completo y sobre el
# entregable real, BGE-M3 solo con top2 gana +0,1074 de NDCG@10 frente a los dos
# encoders con max (p=0,0002, pareado) y cede 0,0213 de F1@3, que es ruido.
DOCUMENT_AGGREGATION = "top2"
MAX_WORDS_PER_FRAGMENT = 250

# Post-filtro de la §8.7: descarta de cada índice los candidatos cuya similitud
# quede por debajo de esta fracción de la mejor de esa consulta en ese índice.
# Relativo y no absoluto porque las escalas de coseno de los dos encoders no son
# comparables. None lo desactiva; el valor sale del barrido, y tiene que ser el
# mismo con el que se generaron los resultados entregados.
RELATIVE_THRESHOLD: float | None = None

# Un fragmento largo se puede partir en varios, cada uno ocupando su posición, o
# recortar. Probamos las dos: partir entrega menos texto (84k palabras por
# corrida frente a 104k) y cubre 6 fragmentos distintos en vez de 10, porque la
# cola de un fragmento de 321 palabras son 71 y se come una posición entera.
SPLIT_LONG_FRAGMENTS = False

# A qué fenómeno corresponde cada consulta, según el orden del archivo.
PHENOMENON_RANGES = {1: (1, 16), 2: (17, 32), 3: (33, 50)}


def phenomenon_of(query_id: str) -> int | None:
    """'q018' -> 2. Devuelve None si el identificador no encaja."""
    try:
        n = int(query_id.lstrip("qQ"))
    except ValueError:
        return None
    for phenomenon, (lo, hi) in PHENOMENON_RANGES.items():
        if lo <= n <= hi:
            return phenomenon
    return None


def set_seeds(seed: int = SEED) -> None:
    """La recuperación es determinista, pero fijamos las semillas igual para que
    algo que se agregue después y traiga aleatoriedad no rompa la reproducción."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --- Consultas -------------------------------------------------------------

_QUERY_ID = re.compile(r"^(q\d{3})\b[\s:.\-]*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str

    @property
    def phenomenon(self) -> int | None:
        return phenomenon_of(self.query_id)


def _queries_from_text(content: str) -> list[Query]:
    """Cada pregunta puede ocupar varias líneas, así que vamos acumulando hasta
    encontrar el siguiente identificador. Cortar por salto de línea truncaría las
    preguntas largas."""
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
    """Acepta PDF, JSONL o texto plano: no está garantizado que el archivo de
    consultas de la evaluación venga en el mismo formato que el de desarrollo."""
    if not path.exists():
        raise FileNotFoundError(f"no se encuentra el archivo de consultas: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        queries = _queries_from_pdf(path)
    elif suffix in (".jsonl", ".json"):
        queries = _queries_from_jsonl(path)
    else:
        queries = _queries_from_text(path.read_text(encoding="utf-8"))

    queries.sort(key=lambda q: q.query_id)
    return queries


# --- Índice vectorial ------------------------------------------------------


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
        # FAISS rellena con -1 si hay menos vecinos de los que se piden.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


def load_index(name: str, root: Path) -> VectorIndex:
    """Carga el índice de un encoder junto con su almacén de metadata.

    FAISS guarda solo los vectores y sus identificadores internos; la metadata va
    aparte. El vínculo es posicional —la línea n describe el vector n— y aquí se
    comprueba en vez de darlo por hecho, porque si se desalinean el sistema
    devuelve textos que no corresponden y nada falla de forma visible.
    """
    import faiss

    folder = root / f"encoder_{name}"
    index_path = folder / "index.faiss"
    metadata_path = folder / "metadata.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"no existe el índice: {index_path}")

    index = faiss.read_index(str(index_path))
    with metadata_path.open(encoding="utf-8") as fh:
        metadata = [json.loads(line) for line in fh if line.strip()]

    if index.ntotal != len(metadata):
        raise ValueError(
            f"índice y metadata desalineados en {name}: "
            f"{index.ntotal} vectores frente a {len(metadata)} registros"
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


# --- Codificación de la consulta -------------------------------------------


class QueryEncoder:
    """Codifica la consulta con el mismo encoder que se usó al indexar.

    Tiene que ser el mismo: con otro, el vector de la consulta y los del índice
    quedarían en espacios distintos. La salida se normaliza para que el producto
    interno del índice equivalga a la similitud coseno.
    """

    def __init__(self, name: str, device: str | None = None):
        if name not in ENCODERS:
            raise KeyError(f"encoder desconocido: {name}")
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
            # Necesario para que IndexFlatIP se comporte como similitud coseno.
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# --- Recuperación ----------------------------------------------------------


@dataclass
class Candidate:
    chunk_id: str
    doc_id: str
    text: str
    fuente: str
    fenomeno: int
    idioma: str = "es"  # del fragmento; decide el segmentador al recortar
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)


def filter_by_relative_threshold(
    ranking: list[tuple[dict, float]], threshold: float | None
) -> list[tuple[dict, float]]:
    """Post-filtro de la §8.7: descarta la cola de baja similitud de un índice.

    El umbral es relativo al mejor candidato de esa consulta en ese índice.
    Como el ranking viene ordenado de mayor a menor, equivale a un k adaptativo
    por consulta y no altera las posiciones de lo que conserva.
    """
    if not threshold or not ranking:
        return ranking

    tope = ranking[0][1]
    if tope <= 0:  # sin señal utilizable, mejor no filtrar nada
        return ranking

    corte = threshold * tope
    return [(meta, sim) for meta, sim in ranking if sim >= corte]


def fuse_rrf(
    rankings: dict[str, list[tuple[dict, float]]], k0: int = RRF_K0
) -> list[Candidate]:
    """Fusiona los rankings de los dos índices con Reciprocal Rank Fusion.

    RRF combina posiciones en lugar de puntuaciones, así que no le afecta que dos
    encoders tengan escalas distintas. Usamos k0 = 60, el valor habitual.

    Con dos encoders densos esa ventaja es menor de lo que parece, porque ambos
    devuelven coseno en el mismo rango; CombSUM podría rendir igual o mejor.
    Queda pendiente compararlos contra el conjunto de evaluación.
    """
    pooled: dict[str, Candidate] = {}

    for encoder, ranking in rankings.items():
        for rank, (meta, _score) in enumerate(ranking, start=1):
            chunk_id = meta["chunk_id"]
            candidate = pooled.get(chunk_id)
            if candidate is None:
                candidate = Candidate(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    text=meta["texto"],
                    fuente=meta.get("fuente", ""),
                    fenomeno=meta.get("fenomeno", 0),
                    idioma=meta.get("idioma") or "es",
                )
                pooled[chunk_id] = candidate
            candidate.score += 1.0 / (k0 + rank)
            candidate.ranks[encoder] = rank

    return sorted(pooled.values(), key=lambda c: c.score, reverse=True)


def apply_phenomenon_boost(
    candidates: list[Candidate], phenomenon: int | None, factor: float = PHENOMENON_BOOST
) -> list[Candidate]:
    """Realza los candidatos del fenómeno esperado, sin excluir a los demás.

    Es un multiplicador suave y no un filtro duro porque varias consultas admiten
    evidencia transversal: q005 y q046 tratan Colombia desde fenómenos distintos,
    y q027 cruza inteligencia artificial con operaciones espaciales. Filtrar en
    firme dejaría fuera documentos legítimamente relevantes.
    """
    if phenomenon is None or factor == 1.0:
        return candidates

    for candidate in candidates:
        if candidate.fenomeno == phenomenon:
            candidate.score *= factor

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def drop_duplicates(candidates: list[Candidate]) -> list[Candidate]:
    """Quita fragmentos con el mismo texto, conservando el mejor puntuado.

    En el corpus hay 4.445 fragmentos (3%) con texto idéntico: tablas reimpresas
    en varios informes, encabezados institucionales repetidos. Dos copias del
    mismo texto valen lo mismo y gastan dos de las diez posiciones para decir una
    sola cosa.

    La comparación va sobre el texto con espacios normalizados, así que también
    caen las variantes que solo difieren en saltos de línea.
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
    """Limita cuántas posiciones puede ocupar un mismo documento.

    Sin el tope, un documento puede llevarse las diez; si resulta no ser
    relevante, se pierde la consulta entera. El tope reparte ese riesgo.
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

    # Si el tope dejó la lista corta, se completa con los mejores descartados.
    if len(selected) < limit:
        taken = {c.chunk_id for c in selected}
        for candidate in candidates:
            if candidate.chunk_id in taken:
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break

    return selected


def document_score(scores: list[float], mode: str) -> float:
    """Colapsa las puntuaciones de los fragmentos de un documento en una.

    - `max`: manda el mejor fragmento. No tiene sesgo de longitud, pero descarta
      la evidencia repetida.
    - `topN`: media de los N mejores. Recompensa que la evidencia esté en varios
      fragmentos buenos sin dejar que cuarenta flojos ganen por acumulación.

    No se admite sumar todos los fragmentos: un documento con cuarenta flojos de
    0,15 acumula 6,0 y desplaza a un informe corto que trae la respuesta exacta
    con 0,85. Con alertas de un párrafo y atlas de cientos de páginas en el mismo
    corpus, ese sesgo sería grave.
    """
    if mode == "max":
        return max(scores)
    if mode.startswith("top") and mode[3:].isdigit():
        best = sorted(scores, reverse=True)[: int(mode[3:])]
        return sum(best) / len(best)
    raise ValueError(
        f"agregación desconocida: {mode!r}. Se admiten max, top2, top3, ..."
    )


def aggregate_to_documents(
    candidates: list[Candidate],
    top: int = TOP_DOCUMENTS,
    mode: str = DOCUMENT_AGGREGATION,
) -> list[str]:
    """Agrega fragmentos a nivel de documento.

    La agregación va por `fuente` y no por `doc_id`: el jurado empareja los
    documentos por `fuente` (§10.2.1), y el corpus tiene 59 nombres repetidos en
    186 archivos — dos doc_id con la misma fuente en el top-3 cuentan como un
    solo acierto posible y desperdician una posición. De cada fuente se reporta
    el doc_id de su mejor fragmento.

    El número de fragmentos solo desempata, con un peso lo bastante pequeño para
    no alterar el orden principal.
    """
    scores: dict[str, list[float]] = {}
    best: dict[str, float] = {}
    holder: dict[str, str] = {}  # fuente -> doc_id del mejor fragmento

    for candidate in candidates:
        key = candidate.fuente or candidate.doc_id
        scores.setdefault(key, []).append(candidate.score)
        previous = best.get(key)
        if previous is None or candidate.score > previous:
            best[key] = candidate.score
            holder[key] = candidate.doc_id

    aggregated = {key: document_score(v, mode) for key, v in scores.items()}
    ordered = sorted(
        aggregated.items(),
        key=lambda kv: (kv[1], len(scores[kv[0]]) * 1e-9),
        reverse=True,
    )
    return [holder[key] for key, _ in ordered[:top]]


class Retriever:
    def __init__(
        self,
        indices: dict[str, VectorIndex],
        k0: int = RRF_K0,
        boost: float = PHENOMENON_BOOST,
        max_per_doc: int = MAX_FRAGMENTS_PER_DOC,
        candidates_per_index: int = CANDIDATES_PER_INDEX,
        threshold: float | None = RELATIVE_THRESHOLD,
        aggregation: str = DOCUMENT_AGGREGATION,
    ):
        if not indices:
            raise ValueError("no hay índices cargados")
        self.indices = indices
        self.k0 = k0
        self.boost = boost
        self.max_per_doc = max_per_doc
        self.candidates_per_index = candidates_per_index
        self.threshold = threshold
        # Se valida al construir y no al escribir la salida: un modo mal escrito
        # debe fallar antes de cargar los modelos, no después de codificar.
        document_score([0.0], aggregation)
        self.aggregation = aggregation
        self._encoders: dict[str, QueryEncoder] = {}

    def _encoder(self, name: str) -> QueryEncoder:
        if name not in self._encoders:
            self._encoders[name] = QueryEncoder(name)
        return self._encoders[name]

    def retrieve(self, query: Query) -> tuple[list[str], list[Candidate]]:
        rankings: dict[str, list[tuple[dict, float]]] = {}

        for name, index in self.indices.items():
            vector = self._encoder(name).encode([query.text])[0]
            hits = index.search(vector, self.candidates_per_index)
            ranking = [(index.metadata[pos], score) for pos, score in hits]
            rankings[name] = filter_by_relative_threshold(ranking, self.threshold)

        candidates = fuse_rrf(rankings, self.k0)
        candidates = apply_phenomenon_boost(candidates, query.phenomenon, self.boost)

        # Diversificar va antes del pooling porque protege la lista de
        # fragmentos, que es lo que mide NDCG@10. La deduplicación se aplica solo
        # ahí: dos documentos distintos pueden compartir un texto y descartar uno
        # lo volvería inalcanzable. La agregación usa el conjunto completo.
        #
        # Se pasan más candidatos de los que se publican: la construcción de la
        # salida puede descartar piezas por el tope por documento, y con
        # repuestos reales a mano no tiene que rellenar repitiendo fragmentos.
        fragments = diversify(
            drop_duplicates(candidates), TOP_FRAGMENTS * 3, self.max_per_doc
        )
        documents = aggregate_to_documents(candidates, TOP_DOCUMENTS, self.aggregation)

        return documents, fragments


# --- Construcción de la salida ---------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[^\s])")

# pysbd trae reglas para estos dos; para el resto —portugués incluido, que pysbd
# no soporta y rechaza con un ValueError— se usa el segmentador inglés, el más
# conservador ante abreviaturas desconocidas. Mismo mapa que usó el fragmentador
# al construir el índice: si divergen, los dos recortan por sitios distintos.
_PYSBD_LANGUAGES = {"es", "en"}


def split_sentences(text: str, language: str = "es") -> list[str]:
    """Segmenta en oraciones para poder recortar sin partir ninguna.

    Usa pysbd con el idioma del fragmento —las abreviaturas que reconoce
    dependen del idioma, y el 87% del corpus está en inglés—, y si no está
    recurre a una expresión regular sobre la puntuación, para que el script siga
    corriendo sin esa dependencia. Las dos respetan los párrafos: nunca unen
    texto de párrafos distintos en una misma oración.
    """
    lang = language if language in _PYSBD_LANGUAGES else "en"
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        try:
            import pysbd

            pieces = pysbd.Segmenter(language=lang, clean=False).segment(paragraph)
        except Exception:
            pieces = _SENTENCE_SPLIT.split(paragraph)
        sentences.extend(p.strip() for p in pieces if p and p.strip())
    return sentences


def word_count(text: str) -> int:
    return len(text.split())


def split_to_limit(
    text: str, max_words: int = MAX_WORDS_PER_FRAGMENT, language: str = "es"
) -> list[str]:
    """Parte un fragmento en piezas de como máximo `max_words`, sin cortar
    ninguna oración por la mitad.

    Si una sola oración ya excede el límite —pasa con tablas volcadas a texto— se
    corta por palabras: respetar el máximo manda, porque un fragmento que lo
    supera se descarta en la evaluación.
    """
    if word_count(text) <= max_words:
        return [text]

    pieces: list[str] = []
    accumulated: list[str] = []
    total = 0

    for sentence in split_sentences(text, language):
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
    """Arma el objeto JSON de una consulta.

    Exactamente 3 documentos y 10 fragmentos, con todos sus campos y ninguno por
    encima de 250 palabras. El `chunk_id` que se reporta sigue siendo el del
    fragmento original del índice aunque se haya recortado: sirve para trazar de
    dónde salió el texto, no para emparejar con el ground truth.

    El tope por documento es firme mientras haya alternativa y cede cuando no la
    hay: si la primera pasada deja posiciones vacías, una segunda las rellena
    con piezas aún no emitidas ignorando el tope. Un fragmento real de un
    documento ya representado informa más que repetir el último para completar
    el esquema, que queda como último recurso.
    """
    docs = [
        {"rank": i, "doc_id": doc_id}
        for i, doc_id in enumerate(documents[:TOP_DOCUMENTS], start=1)
    ]

    # Cuando se parte un fragmento largo, el tope por documento cuenta posiciones
    # entregadas y no fragmentos del índice; si no, un solo documento podría
    # llevarse seis de las diez, que es justo lo que la diversificación evita.
    frags: list[dict] = []
    per_doc: dict[str, int] = {}
    pieces_of: dict[str, list[str]] = {}  # chunk_id -> piezas calculadas una vez
    emitted: dict[str, int] = {}  # chunk_id -> cuántas piezas suyas ya salieron

    def pieces(candidate: Candidate) -> list[str]:
        if candidate.chunk_id not in pieces_of:
            split = split_to_limit(candidate.text, language=candidate.idioma)
            pieces_of[candidate.chunk_id] = split if SPLIT_LONG_FRAGMENTS else split[:1]
        return pieces_of[candidate.chunk_id]

    def emit(candidate: Candidate, respect_cap: bool) -> None:
        for piece in pieces(candidate)[emitted.get(candidate.chunk_id, 0) :]:
            if len(frags) >= TOP_FRAGMENTS:
                return
            if respect_cap and per_doc.get(candidate.doc_id, 0) >= MAX_FRAGMENTS_PER_DOC:
                return
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
            emitted[candidate.chunk_id] = emitted.get(candidate.chunk_id, 0) + 1

    for candidate in fragments:
        if len(frags) >= TOP_FRAGMENTS:
            break
        emit(candidate, respect_cap=True)

    if len(frags) < TOP_FRAGMENTS:
        for candidate in fragments:
            if len(frags) >= TOP_FRAGMENTS:
                break
            emit(candidate, respect_cap=False)

    # The schema demands exactly 3 documents and 10 fragments; a short list is
    # discarded by the automatic evaluator. Padding only triggers on pathological
    # queries where the whole pool runs out of distinct pieces.
    while docs and len(docs) < TOP_DOCUMENTS:
        docs.append({"rank": len(docs) + 1, "doc_id": docs[-1]["doc_id"]})
    while frags and len(frags) < TOP_FRAGMENTS:
        frags.append({**frags[-1], "rank": len(frags) + 1})

    return {"query_id": query_id, "documents": docs, "fragments": frags}


def validate(path: Path, expected_queries: int = 50) -> list[str]:
    """Revisa el archivo antes de entregarlo.

    Un objeto al que le falte un campo, una lista con un número de elementos
    distinto de 3 o 10, o un fragmento de más de 250 palabras, se penaliza o se
    descarta en la evaluación automática. Comprobarlo cuesta nada; equivocarse,
    la entrega entera.
    """
    problems: list[str] = []

    with path.open(encoding="utf-8") as fh:
        lines = [line for line in fh if line.strip()]

    if len(lines) != expected_queries:
        problems.append(f"{len(lines)} líneas, se esperaban {expected_queries}")

    seen: set[str] = set()
    for i, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"línea {i}: JSON inválido ({e})")
            continue

        query_id = record.get("query_id")
        if not query_id:
            problems.append(f"línea {i}: falta query_id")
            continue
        if query_id in seen:
            problems.append(f"línea {i}: query_id repetido ({query_id})")
        seen.add(query_id)

        documents = record.get("documents") or []
        fragments = record.get("fragments") or []

        if len(documents) != TOP_DOCUMENTS:
            problems.append(
                f"{query_id}: {len(documents)} documentos, se esperaban {TOP_DOCUMENTS}"
            )
        if len(fragments) != TOP_FRAGMENTS:
            problems.append(
                f"{query_id}: {len(fragments)} fragmentos, se esperaban {TOP_FRAGMENTS}"
            )

        for fragment in fragments:
            missing = [
                f for f in ("rank", "chunk_id", "doc_id", "text") if f not in fragment
            ]
            if missing:
                problems.append(f"{query_id}: al fragmento le faltan campos {missing}")
                continue
            words = word_count(fragment["text"])
            if words > MAX_WORDS_PER_FRAGMENT:
                problems.append(
                    f"{query_id} rango {fragment['rank']}: {words} palabras "
                    f"(máximo {MAX_WORDS_PER_FRAGMENT})"
                )

    return problems


# --- Entry point ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Genera resultados.jsonl desde el índice.")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    ap.add_argument("--encoders", help="lista separada por comas; por defecto todos los encontrados")
    ap.add_argument("--k0", type=int, default=RRF_K0)
    ap.add_argument("--boost", type=float, default=PHENOMENON_BOOST)
    ap.add_argument("--max-per-doc", type=int, default=MAX_FRAGMENTS_PER_DOC)
    ap.add_argument("--aggregation", default=DOCUMENT_AGGREGATION,
                    help="max o topN (top2, top3): cómo se puntúa un documento "
                         "a partir de sus fragmentos")
    args = ap.parse_args()

    set_seeds()

    queries = load_queries(args.queries)
    print(f"consultas: {len(queries)}")

    found = available_encoders(args.index_root)
    if not found:
        print(f"no hay índices en {args.index_root}")
        return 1

    chosen = [e.strip() for e in args.encoders.split(",")] if args.encoders else found
    missing = set(chosen) - set(found)
    if missing:
        print(f"índices no encontrados: {sorted(missing)} (disponibles: {found})")
        return 1

    indices = {name: load_index(name, args.index_root) for name in chosen}
    for name, index in indices.items():
        print(f"índice {name}: {len(index):,} fragmentos")

    retriever = Retriever(
        indices, k0=args.k0, boost=args.boost, max_per_doc=args.max_per_doc,
        aggregation=args.aggregation
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
        print(f"\n{len(problems)} problemas de formato:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1

    print("formato validado: el esquema es correcto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
