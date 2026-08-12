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

# Entidades reconocidas por fragmento (etapa 04_grafo). Se cachea por la misma
# razón que los embeddings: una pasada de NER sobre el corpus cuesta lo que una
# indexación con encoder base, y no debería repetirse por reconstruir el grafo
# con otra poda. No se versiona, como el resto de `trabajo/`: a diferencia del
# OCR, depende solo de un modelo anclado y se regenera igual en cualquier máquina.
ENTIDADES = TRABAJO / "entidades.jsonl"


def ruta_entidades(backend: str) -> Path:
    """El backend de NER entra en el nombre del archivo, como la huella entra en
    el de la caché de embeddings.

    Sin esto, una corrida de comprobación con `--ner falso` deja una caché que la
    corrida siguiente con el modelo real da por buena, y el grafo del entregable
    sale de un reconocedor de mayúsculas. Es el mismo fallo que evita `huella` en
    `04_indexar`: números plausibles de una configuración que nadie corrió.
    """
    return TRABAJO / f"entidades-{backend}.jsonl"

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


# Cómo se decide dónde empieza y acaba un fragmento. `fijo` acumula oraciones
# hasta agotar el presupuesto de tokens; `jerarquico` toma como unidad la sección
# que declara el documento y solo subdivide las que no caben.
ESTRATEGIA_FIJA = "fijo"
ESTRATEGIA_JERARQUICA = "jerarquico"
ESTRATEGIAS_CHUNKING = (ESTRATEGIA_FIJA, ESTRATEGIA_JERARQUICA)


def clave_chunking(
    chunk: int, solape: float, estrategia: str = ESTRATEGIA_FIJA
) -> str:
    """(504, 0.15) -> 'c504-s015'; con estrategia -> 'c504-s015-jerarquico'.

    Identifica qué fragmentación produjo unos artefactos. Vive aquí y no en el
    barrido porque la usan tanto quien escribe los índices como quien decide si
    ya existen, y dos versiones de esta cadena harían que la caché no acertara.

    La estrategia entra en la clave porque dos fragmentaciones del mismo tamaño
    y solape producen fragmentos **distintos** si una parte por secciones. Sin
    esto, pedir la jerárquica a 504/0,15 encontraría cacheados los fragmentos
    fijos y los daría por buenos: números plausibles de una configuración que
    nadie corrió, que es el peor fallo que puede tener un barrido.

    `fijo` no lleva sufijo a propósito: es lo que ya está escrito en disco y
    renombrarlo invalidaría todos los índices de `pruebas/` sin ganar nada.
    """
    base = f"c{chunk}-s{int(round(solape * 100)):03d}"
    return base if estrategia == ESTRATEGIA_FIJA else f"{base}-{estrategia}"


def ruta_fragmentos_prueba(
    chunk: int, solape: float, estrategia: str = ESTRATEGIA_FIJA
) -> Path:
    return PRUEBAS_FRAGMENTOS / f"{clave_chunking(chunk, solape, estrategia)}.jsonl"


def ruta_indices_prueba(
    chunk: int, solape: float, estrategia: str = ESTRATEGIA_FIJA
) -> Path:
    return PRUEBAS_INDICES / clave_chunking(chunk, solape, estrategia)

# Entregable
ENTREGA = RAIZ / "entrega"
BASE_VECTORIAL = ENTREGA / "base_vectorial"
RESULTADOS = ENTREGA / "resultados.jsonl"

# El grafo de conocimiento va dentro de `base_vectorial/`, no al lado: es donde
# lo coloca el árbol de directorios de la §1.4, a la misma altura que las
# carpetas por encoder. El nombre del archivo también lo fija el reto.
GRAFO = BASE_VECTORIAL / "grafo" / "grafo.graphml"

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
ENCODERS_ENTREGA = ("bge-m3",)

# Cómo se puntúa un documento a partir de sus fragmentos: `max` toma el mejor,
# `topN` promedia los N mejores.
#
# **Medido**: sobre el corpus completo y el entregable real, bge-m3 solo con
# top2 gana +0,1074 de NDCG@10 frente a los dos encoders con max (p=0,0002,
# pareado) y cede 0,0213 de F1@3, que es ruido (p=0,51). Recupera además el
# F1@3 del fenómeno 2, que es donde `max` más perdía (0,7708 -> 0,8750).
AGREGACION_DOCUMENTOS = "top2"

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

# --- Chunking jerárquico --------------------------------------------------
# La estrategia `jerarquico` toma la sección del documento como unidad en vez
# del presupuesto de tokens. Estos tres umbrales deciden qué hacer con cada una.
#
# Una sección por debajo de MIN se entrega entera aunque sobre presupuesto: la
# apuesta de esta estrategia es que una sección corta y completa recupera mejor
# que media sección de tamaño óptimo. Por encima de MAX se subdivide con el
# mismo `agrupar` de la estrategia fija, así que sigue sin partir una oración.
# Entre las dos se deja entera, que es lo que significa respetar la estructura.
SECCION_MIN_TOKENS = 450
SECCION_MAX_TOKENS = 650

# Por debajo de esto una sección no se sostiene sola —suele ser el encabezado
# suelto, o el pie de una sección que empieza en la página siguiente— y se
# acumula con la que sigue. Sin esta regla, un documento con cuarenta epígrafes
# cortos mete cuarenta fragmentos de dos líneas en el índice, y cada uno compite
# por una de las diez posiciones que evalúa NDCG@10.
SECCION_MIN_AUTONOMA = 60

# Un fragmento jerárquico llega hasta SECCION_MAX_TOKENS, por encima del
# presupuesto de la estrategia fija. Es inocuo con encoders de ventana larga
# (BGE-M3 tiene 8192) y trunca en silencio con los de 512, así que las recetas
# que pidan esta estrategia se comprueban contra este techo y no contra `chunk`.


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
# **Medido.** Barriendo 25/50/100/200 con la fusión convexa, el F1@3 sube de
# forma monótona —0,533, 0,587, 0,628, 0,676— y bajar a 25 es la única pérdida
# que se acerca a ser significativa (p=0,004 pareado). Más allá de 200 la curva
# se aplana: 400, 800, 1600 y 3200 no mejoran nada (el mejor puntual, 400 con
# F1 0,689, da p=0,68). Así que 200 no es una cifra heredada, es donde satura.
CANDIDATOS_POR_INDICE = 200
RRF_K0 = 60

# **Medido y sin efecto sobre el F1@3**, que es exactamente lo que cabía esperar:
# la diversificación recorta la lista de fragmentos y la agregación a documento
# trabaja sobre el pool completo, así que d=1 y d=10 dan el mismo 0,676. Sobre el
# NDCG@10 el efecto es leve y no significativo (0,431 con d=1, 0,455 con d=10).
# Se deja en 3 porque es lo que protege el top-10 de que un documento lo acapare,
# que es para lo que existe.
MAX_FRAGMENTOS_POR_DOC = 3  # diversificación del top-10

# **Medido y sin efecto**: 1,0 / 1,05 / 1,15 / 1,3 / 1,6 quedan todos dentro del
# ruido (el mejor, 1,3, da p=0,30 en NDCG@10).
#
# El desglose explica por qué no puede ayudar mucho: subir el realce a 1,3
# beneficia al fenómeno 2 (+0,020 NDCG) y perjudica al 1 (-0,011), y el 1 es
# justo el que peor rinde. Pero esa brecha —NDCG 0,311 en el fenómeno 1 contra
# 0,575 en el 2— no es del sistema sino del corpus: el fenómeno 1 tiene 6,4
# documentos relevantes por consulta contra 9,9 del 2, y 15,3 fragmentos con
# relevancia positiva contra 21,8. Hay menos que encontrar, así que realzar no
# lo arregla.
BOOST_FENOMENO = 1.05  # multiplicador suave, no filtro duro

# Post-filtro de la §8.7: descarta de cada índice los candidatos cuya similitud
# quede por debajo de esta fracción de la mejor de esa consulta en ese índice.
# Es relativo y no absoluto porque las escalas de coseno de BGE-M3 y mE5 no son
# comparables (mE5 comprime todo hacia 0,75–0,92). None lo desactiva; el valor
# se decide en el barrido, no a priori.
UMBRAL_RELATIVO: float | None = None

# Realimentación de pseudo-relevancia (Rocchio). `PRF_K` primeros resultados se
# promedian y su centroide desplaza el vector de consulta con peso `PRF_BETA`
# antes de una segunda búsqueda. `PRF_K = 0` lo desactiva.
#
# **Medido y descartado**, no pendiente. Se probaron k en {3, 5, 10} y beta en
# {0,3, 0,5, 0,8} sobre la submuestra, contra el mismo sistema sin expansión y
# con el test pareado de `metricas.p_valor_permutacion`:
#
#   - con BGE-M3 solo, el mejor (k=10, beta=0,3) sube el NDCG@10 de 0,4879 a
#     0,5034 con p=0,047, que no sobrevive a la corrección por las catorce
#     comparaciones de la rejilla (0,05/14 = 0,0036). Es el máximo de una
#     búsqueda de hiperparámetros, no un efecto.
#   - con la fusión convexa **hace daño** al F1@3 en las ocho configuraciones,
#     de 0,6693 a entre 0,613 y 0,641, con p=0,026 en la peor.
#
# La lectura es que los primeros resultados de este corpus no son lo bastante
# limpios como para que su centroide oriente la consulta: la deriva de tema pesa
# más que el vocabulario que aporta. Queda implementado porque el experimento
# tiene que ser repetible sobre el corpus completo, donde la densidad de
# relevantes es cinco veces menor y el resultado puede no ser el mismo.
PRF_K = 0
PRF_BETA = 0.5

# Correspondencia consulta -> fenómeno, según el orden del archivo de preguntas.
RANGOS_FENOMENO = {1: (1, 16), 2: (17, 32), 3: (33, 50)}

# --- Grafo de conocimiento (§7, bonus) ------------------------------------
# Ninguna de estas constantes está medida todavía: el grafo se construye pero no
# se conecta a la recuperación, así que no hay NDCG@10 contra el que calibrarlas.
# Los valores son puntos de partida razonados, y el comentario dice el
# razonamiento para que quien lo mida sepa qué esperaba encontrar.

# Entidades por fragmento en el grafo. Un fragmento de 504 tokens menciona
# muchas más, pero las que están por debajo del tope son las que aparecen una
# vez de pasada: engordan el XML —que es donde duele, GraphML es texto— sin
# abrir caminos que no abra ya la entidad principal.
GRAFO_MAX_ENTIDADES_POR_FRAGMENTO = 12

# Poda por frecuencia documental. Por abajo, una entidad que aparece en un solo
# documento no conecta dos fragmentos, que es lo único que sabe hacer el canal
# del grafo. Por arriba, una que aparece en más del 5% del corpus es membrete
# —«Colombia» en un corpus colombiano, «AI» en el fenómeno 1— y devolvería medio
# índice sin discriminar nada.
GRAFO_MIN_DF = 2
GRAFO_MAX_FRACCION_DF = 0.05

# Cuántos fragmentos aporta el canal del grafo antes de fusionar. Deliberadamente
# menor que CANDIDATOS_POR_INDICE (200): aquel está medido para índices densos,
# donde la cola profunda solo suma consenso al F1@3, mientras que este canal
# ordena por conteo de menciones y su cola es mucho más ruidosa. Empezar bajo y
# subir si el barrido lo pide es más seguro que al revés.
GRAFO_CANDIDATOS = 50

# Cuánto puntúa un fragmento que menciona a un vecino de primer orden de la
# entidad consultada, frente a uno que la menciona directamente. Los vecinos son
# lo que distingue un grafo de una lista invertida —la consulta sobre basura
# orbital debería alcanzar las misiones que la generan—, pero a peso completo
# ahogarían la señal directa. 0 desactiva el salto y deja el canal en búsqueda
# por entidad exacta, que es la comparación de control del barrido.
GRAFO_PESO_VECINO = 0.35

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
