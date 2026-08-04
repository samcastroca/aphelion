"""Rutas y constantes del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

# Insumos provistos por ADL. El corpus (3 GB) no se versiona; el inventario, las
# preguntas y la especificación sí, porque el pipeline no arranca sin ellos.
DATOS = RAIZ / "data"
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

# Texto de los fragmentos juzgados, indexado por chunk_id. Se versiona junto al
# ground truth porque sin él los juicios solo sirven para la fragmentación que
# los originó: al cambiar el tamaño de chunk los `chunk_id` dejan de existir y la
# relevancia hay que emparejarla por texto, que es además como lo hará el jurado
# (§10.2.1).
GROUND_TRUTH_TEXTOS = DATOS / "ground_truth_textos.jsonl"

# Subconjunto de documentos para iterar rápido. Se versiona: comparar dos
# configuraciones solo tiene sentido si ambas vieron el mismo corpus.
SUBMUESTRA = DATOS / "submuestra.json"

# Documentación. El informe se redacta en español y se exporta a PDF a la entrega.
DOCS = RAIZ / "docs"

# Artefactos intermedios (no se versionan)
TRABAJO = RAIZ / "trabajo"
TEXTO_CRUDO = TRABAJO / "texto"  # un .json por documento extraído
ONNX = TRABAJO / "onnx"  # modelos exportados para la GPU Radeon
FRAGMENTOS = TRABAJO / "fragmentos.jsonl"
CONSULTAS = TRABAJO / "consultas.jsonl"

# Experimentación. Los índices y los fragmentos se comparten entre experimentos
# porque son caros y su contenido queda determinado por (chunking, encoder): dos
# experimentos que pidan el mismo índice deben reutilizarlo, no duplicar 117 MB.
# Las métricas, en cambio, van por experimento, que es lo que se compara.
#
#   pruebas/fragmentos/c504-s015.jsonl        compartido
#   pruebas/indices/c504-s015/encoder_X/      compartido
#   pruebas/<nombre>/metricas.jsonl           por experimento
#   pruebas/<nombre>/resumen.txt              por experimento
#   pruebas/<nombre>/config.json              por experimento
PRUEBAS = RAIZ / "pruebas"
PRUEBAS_FRAGMENTOS = PRUEBAS / "fragmentos"
PRUEBAS_INDICES = PRUEBAS / "indices"


def clave_chunking(chunk: int, solape: float) -> str:
    """(504, 0.15) -> 'c504-s015'.

    Identifica qué fragmentación produjo unos artefactos. Vive aquí y no en el
    barrido porque la usan tanto quien escribe los índices como quien decide si
    ya existen, y dos versiones de esta cadena harían que la caché no acertara.
    """
    return f"c{chunk}-s{int(round(solape * 100)):03d}"


def ruta_fragmentos_prueba(chunk: int, solape: float) -> Path:
    return PRUEBAS_FRAGMENTOS / f"{clave_chunking(chunk, solape)}.jsonl"


def ruta_indices_prueba(chunk: int, solape: float) -> Path:
    return PRUEBAS_INDICES / clave_chunking(chunk, solape)

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
    # --- Candidatos para el barrido -----------------------------------------
    # Todos son arquitecturas **encoder** con licencia permisiva, que es lo que
    # el reto exige (§4.2 prohíbe los decoders y §4.3 pide licencia libre).
    # Quedan fuera por licencia Jina v3 (CC-BY-NC-4.0) y por arquitectura
    # cualquier derivado de un backbone autoregresivo, como Qwen3-Embedding o
    # e5-mistral, por más que lideren MTEB.
    #
    # `dim` se verifica al construir el índice —`vectores.construir` compara la
    # forma real de la matriz— así que un valor equivocado aquí falla ruidoso y
    # no en silencio.
    "me5-base": {
        # La mitad de dimensión y un tercio de parámetros que me5-large. Entra
        # porque el barrido necesita al menos un encoder que se codifique rápido:
        # con él, probar cuatro tamaños de chunk cuesta lo que uno de los grandes.
        "modelo": "intfloat/multilingual-e5-base",
        "dim": 768,
        "max_tokens": 512,
        "prefijo_consulta": "query: ",
        "prefijo_pasaje": "passage: ",
        "pooling": "mean",
    },
    "me5-small": {
        # El más barato del catálogo. Sirve para barrer la rejilla de chunking en
        # minutos y quedarse solo con los tamaños prometedores para los grandes.
        "modelo": "intfloat/multilingual-e5-small",
        "dim": 384,
        "max_tokens": 512,
        "prefijo_consulta": "query: ",
        "prefijo_pasaje": "passage: ",
        "pooling": "mean",
    },
    "me5-large-instruct": {
        # Mismo backbone que me5-large (XLM-RoBERTa large) afinado con
        # instrucciones. El prefijo de consulta lleva la tarea descrita en
        # lenguaje natural, y los pasajes van sin prefijo — al revés que la
        # familia e5 clásica. Interesa porque la asimetría instruida suele ayudar
        # justo en lo que aquí falla: consultas largas en español contra pasajes
        # en inglés.
        "modelo": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "max_tokens": 512,
        "prefijo_consulta": (
            "Instruct: Given a question in Spanish, retrieve passages in any "
            "language that answer it\nQuery: "
        ),
        "prefijo_pasaje": "",
        "pooling": "mean",
    },
    "gte-multilingual-base": {
        # Familia distinta de las dos anteriores (no es un XLM-R afinado por
        # intfloat ni por BAAI), con ventana larga y solo 768 dimensiones. Es el
        # candidato con más probabilidad de aportar algo *nuevo* a la fusión:
        # cuanto menos se parezca su espacio al de BGE-M3, más consenso
        # informativo puede añadir el RRF.
        "modelo": "Alibaba-NLP/gte-multilingual-base",
        "dim": 768,
        "max_tokens": 8192,
        "prefijo_consulta": "",
        "prefijo_pasaje": "",
        "pooling": "cls",
        # Su implementación vive en el repositorio del modelo, no en
        # transformers, así que hay que autorizarla explícitamente.
        "trust_remote_code": True,
    },
    "labse": {
        # **Control, no candidato.** El informe afirma que LaBSE no sirve aquí
        # porque se entrenó para alinear pares de frases paralelas y carece de
        # noción de relevancia temática asimétrica. Esa afirmación está tomada de
        # la literatura y nunca se midió sobre este corpus. Incluirlo en el
        # barrido la convierte en un número: si LaBSE queda último por mucho, la
        # justificación del informe queda respaldada con datos propios.
        "modelo": "sentence-transformers/LaBSE",
        "dim": 768,
        "max_tokens": 256,
        "prefijo_consulta": "",
        "prefijo_pasaje": "",
        "pooling": "cls",
    },
}

# Los que se indexan cuando nadie pide otra cosa. El resto del catálogo existe
# para el barrido y no entra en la entrega salvo que el barrido lo justifique.
ENCODERS_ENTREGA = ("bge-m3", "me5-large")

ENCODER_PRINCIPAL = "bge-m3"

# --- Chunking -------------------------------------------------------------

CHUNK_TOKENS = 512  # ventana del modelo más corto (mE5-large)

# La ventana de 512 de mE5 incluye lo que el fragmento no trae: los tokens
# especiales (<s>, </s>) y el prefijo "passage: " con que E5 exige codificar los
# pasajes. Un fragmento presupuestado a 512 tokens de contenido entra al modelo
# con ~518 y pierde la cola en silencio. La reserva descuenta ese sobrecoste del
# presupuesto de fragmentación; la ventana del modelo (y del grafo ONNX) sigue
# siendo CHUNK_TOKENS.
RESERVA_TOKENS_ENCODER = 8
CHUNK_PRESUPUESTO = CHUNK_TOKENS - RESERVA_TOKENS_ENCODER

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

# Post-filtro de la §8.7: descarta de cada índice los candidatos cuya similitud
# quede por debajo de esta fracción de la mejor de esa consulta en ese índice.
# Es relativo y no absoluto porque las escalas de coseno de BGE-M3 y mE5 no son
# comparables (mE5 comprime todo hacia 0,75–0,92). None lo desactiva; el valor
# se decide en el barrido, no a priori.
UMBRAL_RELATIVO: float | None = None

# Correspondencia consulta -> fenómeno, según el orden del archivo de preguntas.
RANGOS_FENOMENO = {1: (1, 16), 2: (17, 32), 3: (33, 50)}

# --- Extracción -----------------------------------------------------------

UMBRAL_PDF_SIN_TEXTO = 200  # caracteres; por debajo se considera escaneado


def cabe_en_ventana(chunk: int, encoder: str) -> bool:
    """¿Un fragmento de `chunk` tokens entra completo en ese encoder?

    Codificar un fragmento más largo que la ventana del modelo no falla: el
    tokenizador lo trunca y devuelve un vector que no representa la cola. Es el
    peor fallo posible en un barrido, porque produce números plausibles para una
    configuración que nadie está midiendo. Se comprueba contra `max_tokens` más
    la reserva de tokens especiales y prefijo, que es lo que gasta el encoder
    por encima del contenido.
    """
    return chunk + RESERVA_TOKENS_ENCODER <= ENCODERS[encoder]["max_tokens"]


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
