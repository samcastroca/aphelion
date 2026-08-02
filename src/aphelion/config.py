"""Rutas y constantes del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Insumos provistos por ADL. El corpus (3 GB) no se versiona; el inventario, las
# preguntas y la especificación sí, porque el pipeline no arranca sin ellos.
DATOS = RAIZ / "datos"
CORPUS = DATOS / "CORPUS CODEFEST AD ASTRA 2026"
INVENTARIO = DATOS / "Indice_Datos_Codefest.xlsx"
PREGUNTAS_PDF = DATOS / "Extracto_Preguntas_50_v2.pdf"
ESPECIFICACION = DATOS / "CODEFEST_2026-1 Especificacion.pdf"

# Texto reconocido de los PDFs escaneados. Se versiona —pesa 1,7 MB— porque es
# el único artefacto del pipeline que depende de software externo (Tesseract con
# su paquete de español) y cuesta horas de CPU. Cachearlo evita repetirlo en cada
# máquina donde se ejecute el proyecto.
OCR_CACHE = DATOS / "ocr.jsonl"

# Evaluación interna. Se versiona por la misma razón que el OCR: son juicios
# emitidos a mano por cuatro personas, el artefacto más caro del proyecto y el
# único que no se puede regenerar ejecutando nada.
ANOTACION = DATOS / "anotacion"  # un CSV por anotador
GROUND_TRUTH = DATOS / "ground_truth.jsonl"

# Documentación. El informe se redacta en español y se exporta a PDF a la entrega.
DOCS = RAIZ / "docs"

# Artefactos intermedios (no se versionan)
TRABAJO = RAIZ / "trabajo"
TEXTO_CRUDO = TRABAJO / "texto"  # un .json por documento extraído
ONNX = TRABAJO / "onnx"  # modelos exportados para la GPU Radeon
FRAGMENTOS = TRABAJO / "fragmentos.jsonl"
CONSULTAS = TRABAJO / "consultas.jsonl"

# Entregable
ENTREGA = RAIZ / "entrega"
BASE_VECTORIAL = ENTREGA / "base_vectorial"
RESULTADOS = ENTREGA / "resultados.jsonl"

# --- Encoders -------------------------------------------------------------
# Ambos son arquitecturas encoder (XLM-RoBERTa) con licencia MIT. El reto
# prohíbe los decoders para generar embeddings, lo que descarta Qwen3-Embedding
# pese a liderar MTEB.

ENCODERS = {
    "bge-m3": {
        "modelo": "BAAI/bge-m3",
        "dim": 1024,
        "max_tokens": 8192,
        "prefijo_consulta": "",
        "prefijo_pasaje": "",
        # BGE-M3 toma el token CLS. Usar mean pooling por descuido baja el coseno
        # contra la referencia de 0.999999 a 0.81: parece un problema numérico y
        # es un error de pooling.
        "pooling": "cls",
    },
    "me5-large": {
        "modelo": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "max_tokens": 512,
        # E5 se entrenó con prefijos asimétricos; omitirlos degrada la recuperación.
        "prefijo_consulta": "query: ",
        "prefijo_pasaje": "passage: ",
        # E5 promedia los tokens con la máscara de atención. No es intercambiable
        # con el CLS de BGE-M3.
        "pooling": "mean",
    },
}

ENCODER_PRINCIPAL = "bge-m3"

# --- Chunking -------------------------------------------------------------

CHUNK_TOKENS = 512
CHUNK_SOLAPE = 0.15
MIN_TOKENS_FRAGMENTO = 10  # por debajo son restos de tabla, no contenido
MAX_PALABRAS_FRAGMENTO = 250  # límite que impone el reto

# Un fragmento largo se puede dividir en sub-fragmentos, cada uno ocupando su
# propia posición. El argumento a favor era que truncar descarta evidencia: el 76,5%
# de los fragmentos excede las 250 palabras.
#
# Medido, el argumento no se sostiene. Subdividir **reduce** el texto entregado
# de 104k a 84k palabras por corrida y la cobertura de 10 fragmentos distintos a
# 6, porque las colas cortas —la segunda pieza de un fragmento de 321 palabras
# son 71— ocupan un rango entero con poco contenido, mientras que truncar deja
# las diez posiciones con 250 palabras cada una.
#
# Queda implementado y desactivado: si el ground truth muestra que la cola de un
# fragmento bien posicionado vale más que un fragmento nuevo peor posicionado, se
# activa desde el barrido. Hasta entonces manda lo medido.
SUBDIVIDIR_FRAGMENTOS = False

# Tope de fragmentos por documento tabular. Medido: 30 archivos CSV/XLSX aportan
# 91.412 de los 149.571 fragmentos (61% del índice), y tres exportaciones
# bibliográficas de PubMed solas ocupan el 51,5%. Son listados de referencias
# biomédicas ajenas a las 50 consultas: consumen más de la mitad del tiempo de
# codificación y compiten en cada búsqueda contra la evidencia útil.
#
# El tope se aplica solo a formatos tabulares, donde cada fila es un registro
# independiente y el valor marginal de la fila 3.000 es nulo. Los PDF largos y
# legítimos —una ley de 1.960 fragmentos— no se tocan.
#
# No se excluyen los archivos: el emparejamiento del F1@3 es por `fuente`, así
# que el documento debe seguir siendo alcanzable.
MAX_FRAGMENTOS_TABULARES = 400
FORMATOS_TABULARES = frozenset({"csv", "xlsx"})

# --- Recuperación ---------------------------------------------------------

TOP_DOCUMENTOS = 3
TOP_FRAGMENTOS = 10
# Cuántos candidatos trae cada índice antes de fusionar. 100 es el mínimo
# razonable y se recomienda entre 200 y 500 para recuperación de propósito
# general. Lo que decide es que un documento que no aparezca en este pool no lo
# puede recuperar ninguna etapa posterior, y a nosotros nos importa el pool
# profundo sobre todo para el F1@3: un documento que queda en el puesto 95 de un
# índice y en el 120 del otro solo suma consenso si los dos lo traen.
#
# Ampliarlo no desplaza nada del top-10: con k0=60, un candidato en el puesto 150
# aporta 1/210 frente a 1/65 de uno en el puesto 5. Y la búsqueda es exhaustiva
# de todos modos, así que no cuesta tiempo.
CANDIDATOS_POR_INDICE = 200
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
