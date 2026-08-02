"""Rutas y constantes del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

CORPUS = RAIZ / "CORPUS CODEFEST AD ASTRA 2026"
INVENTARIO = RAIZ / "Indice_Datos_Codefest.xlsx"
PREGUNTAS_PDF = RAIZ / "Extracto_Preguntas_50_v2.pdf"

# Artefactos intermedios (no se versionan)
TRABAJO = RAIZ / "trabajo"
TEXTO_CRUDO = TRABAJO / "texto"  # un .json por documento extraído
FRAGMENTOS = TRABAJO / "fragmentos.jsonl"
CONSULTAS = TRABAJO / "consultas.jsonl"
GROUND_TRUTH = TRABAJO / "ground_truth.jsonl"

# Entregable
ENTREGA = RAIZ / "entrega"
BASE_VECTORIAL = ENTREGA / "base_vectorial"
RESULTADOS = ENTREGA / "resultados.jsonl"

# --- Encoders -------------------------------------------------------------
# Ambos son arquitecturas encoder (XLM-RoBERTa) con licencia MIT. La
# especificación prohíbe los decoders para generar embeddings (§4.2), lo que
# descarta Qwen3-Embedding pese a liderar MTEB.

ENCODERS = {
    "bge-m3": {
        "modelo": "BAAI/bge-m3",
        "dim": 1024,
        "max_tokens": 8192,
        "prefijo_consulta": "",
        "prefijo_pasaje": "",
    },
    "me5-large": {
        "modelo": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_tokens": 512,
        # E5 se entrenó con prefijos asimétricos; omitirlos degrada la recuperación.
        "prefijo_consulta": "query: ",
        "prefijo_pasaje": "passage: ",
    },
}

ENCODER_PRINCIPAL = "bge-m3"

# --- Chunking -------------------------------------------------------------

CHUNK_TOKENS = 512
CHUNK_SOLAPE = 0.15
MIN_TOKENS_FRAGMENTO = 10  # por debajo son restos de tabla, no contenido
MAX_PALABRAS_FRAGMENTO = 250  # límite de la especificación, §9.2

# --- Recuperación ---------------------------------------------------------

TOP_DOCUMENTOS = 3
TOP_FRAGMENTOS = 10
CANDIDATOS_POR_INDICE = 100
RRF_K0 = 60
MAX_FRAGMENTOS_POR_DOC = 3  # diversificación del top-10
BOOST_FENOMENO = 1.05  # multiplicador suave, no filtro duro

# Correspondencia consulta -> fenómeno, según el orden del archivo de preguntas.
RANGOS_FENOMENO = {1: (1, 16), 2: (17, 32), 3: (33, 50)}

# --- Extracción -----------------------------------------------------------

UMBRAL_PDF_SIN_TEXTO = 200  # caracteres; por debajo se considera escaneado


def fenomeno_de_consulta(query_id: str) -> int | None:
    """'q018' -> 2. Devuelve None si el identificador no encaja."""
    try:
        n = int(query_id.lstrip("q"))
    except ValueError:
        return None
    for fenomeno, (lo, hi) in RANGOS_FENOMENO.items():
        if lo <= n <= hi:
            return fenomeno
    return None
