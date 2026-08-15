"""Etapa 1 del grafo: reconocimiento de entidades y canonicalización.

**Por qué NER de tipos abiertos.** El corpus no es de noticias: las entidades que
importan a estas 50 consultas son *sistema de armas autónomo*, *órbita baja
terrestre*, *política pública de IA*, *observatorio*, además de los países y
organizaciones de siempre. Un clasificador con las cuatro clases de CoNLL
(PER/ORG/LOC/MISC) mete todo eso en MISC o lo pierde, así que se usa un modelo
zero-shot al que se le pasan los tipos como etiquetas.

**Por qué las etiquetas van en inglés.** El modelo es multilingüe en el texto de
entrada —declara es/en/pt, que son justo los tres idiomas del corpus— pero sus
prompts de entrenamiento son mayoritariamente ingleses, y una etiqueta en inglés
recupera igual sobre texto español. Los tipos del grafo sí van en español, que
es la convención del proyecto: `TIPOS` traduce entre ambos.

**Licencias.** Los dos backends soportados son Apache-2.0
(`fastino/gliner2-multi-v1` y su export ONNX, `urchade/gliner_multi-v2.1`), lo
que cumple la §4.3 sin discusión. Quedan fuera por licencia no comercial
`Babelscape/wikineural-multilingual-ner` (CC BY-NC-SA 4.0), que es el modelo
multilingüe que uno elegiría por defecto, y los modelos de spaCy en español
(GPL-3.0, heredada de UD AnCora).

**Por qué el backend por defecto es `gliner2` y no el export ONNX.** Los dos
sirven el mismo modelo y los mismos pesos; lo que cambia es el motor. El paquete
oficial `gliner2` coloca el modelo en la GPU (`map_location`) y expone
`batch_extract_entities`, que infiere sobre una lista de textos de una vez. El
binding Python de `gliner2-onnx` no tiene ninguna de las dos cosas: solo
`extract_entities(text, labels)`, y con `providers=None` —que es su defecto—
abre la sesión con `CPUExecutionProvider` sin avisar de que la GPU no se está
usando. Sobre el corpus entero eso son ~160.000 inferencias sueltas en CPU: **15
horas medidas, frente a los minutos que cuesta la misma pasada por lotes en una
GPU.** `NerOnnx` se conserva porque es el único camino a DirectML, que es la GPU
de la máquina de desarrollo.

El conflicto de dependencias que dejó fuera a `gliner` v1 —declara
`transformers<5.14.0` y el proyecto fija `>=5.14.1`, el mismo choque que ya
excluyó a `optimum`— no aplica a `gliner2`: su extra `local` pide `torch` y
`transformers` sin tope de versión, así que convive con el entorno del proyecto y
no hace falta correr esta etapa aislada.

**El resultado se cachea.** Una pasada sobre los 64.484 fragmentos cuesta lo que
una indexación con encoder base, así que se escribe a `trabajo/entidades.jsonl` y
las corridas siguientes la leen. Es la misma razón por la que existe la caché de
embeddings, y el mismo trato: `trabajo/` no se versiona.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Tipo del grafo (español, convención del proyecto) -> etiqueta que se le pasa al
# modelo (inglés, que es como se entrenaron sus prompts).
#
# El inventario es cerrado y corto a propósito. Cada etiqueta adicional multiplica
# el coste de la pasada —el modelo puntúa cada span contra cada etiqueta— y
# diluye el grafo con nodos que ninguna consulta va a tocar. Estos nueve cubren
# el vocabulario de los tres fenómenos: IA y defensa (1), seguridad espacial (2),
# dinámicas territoriales (3).
TIPOS: dict[str, str] = {
    "organizacion": "organization",
    "pais": "country",
    "lugar": "location",
    "persona": "person",
    "tecnologia": "technology",
    "sistema_militar": "weapon system",
    "objeto_espacial": "spacecraft or orbital object",
    "politica": "policy or regulation",
    "evento": "event",
}

# Por debajo de esto la mención es ruido. No está medido sobre este corpus: es el
# punto donde los modelos GLiNER dejan de proponer spans defendibles según su
# tarjeta. Al calibrarlo, la señal a mirar no es la precisión del NER sino el
# NDCG@10 del canal del grafo en el barrido, que es lo que acaba importando.
UMBRAL_CONFIANZA = 0.55

# Una entidad de una sola letra o de cincuenta palabras no es una entidad: la
# primera es una viñeta mal segmentada y la segunda, media frase.
MIN_CARACTERES = 2
MAX_PALABRAS = 8

_NO_ALFANUM = re.compile(r"[^\w\s]+", re.UNICODE)
_ESPACIOS = re.compile(r"\s+")

# Artículos y determinantes de los tres idiomas del corpus. «the European Union»
# y «la Unión Europea» tienen que colapsar al mismo nodo que «European Union» y
# «Unión Europea», o el grafo duplica cada entidad tantas veces como formas de
# citarla tenga el texto.
_ARTICULOS = frozenset(
    {"el", "la", "los", "las", "un", "una", "unos", "unas",
     "the", "a", "an",
     "o", "os", "as", "um", "uma", "uns", "umas"}
)


@dataclass(frozen=True)
class Mencion:
    """Una entidad detectada en un fragmento concreto.

    Guarda el fragmento y el documento porque la §7.2 exige que cada tripleta
    conserve su procedencia, y las tripletas se arman a partir de menciones.
    """

    chunk_id: str
    doc_id: str
    texto: str  # tal como aparece en el fragmento
    tipo: str  # clave de TIPOS
    inicio: int  # desplazamiento en caracteres dentro del fragmento
    fin: int
    confianza: float

    @property
    def clave(self) -> str:
        """Forma normalizada; dos menciones con la misma clave son la misma entidad."""
        return normalizar_nombre(self.texto)


def normalizar_nombre(nombre: str) -> str:
    """Minúsculas, sin tildes ni puntuación y sin artículo inicial.

    No se reutiliza `evaluacion.emparejamiento.normalizar` aunque se le parezca:
    aquel normaliza pasajes para medir solape de n-gramas y no toca artículos
    —hacerlo alteraría el conteo—, mientras que aquí el artículo es justo lo que
    hay que quitar para que dos menciones de la misma entidad colapsen.
    """
    nombre = unicodedata.normalize("NFKD", nombre.lower())
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    nombre = _NO_ALFANUM.sub(" ", nombre)
    palabras = _ESPACIOS.sub(" ", nombre).strip().split()
    if len(palabras) > 1 and palabras[0] in _ARTICULOS:
        palabras = palabras[1:]
    return " ".join(palabras)


def admisible(texto: str) -> bool:
    """Filtro de forma, previo a cualquier decisión del modelo."""
    limpio = texto.strip()
    if len(limpio) < MIN_CARACTERES or len(limpio.split()) > MAX_PALABRAS:
        return False
    # Sin una letra no hay nombre: son cifras de tabla, folios y códigos sueltos,
    # que en un corpus con 30 archivos tabulares aparecen a millares.
    return any(c.isalpha() for c in limpio)


# --- Ventaneo -------------------------------------------------------------

# En caracteres, no en tokens: aquí no hay tokenizador y no hace falta, porque lo
# que se busca es quedarse holgadamente por debajo de la ventana del modelo (384
# tokens, frente a los 504 del fragmento). 1200 caracteres son ~300 tokens en
# inglés y menos en español.
VENTANA = 1200
SOLAPE = 200


def ventanas(texto: str, ancho: int = VENTANA, solape: int = SOLAPE) -> list[tuple[int, int]]:
    """Los tramos `(inicio, fin)` en que se recorre un fragmento.

    Devuelve posiciones y no los textos: con 64.484 fragmentos, materializar los
    ~160.000 trozos son 190 MB duplicados de lo que ya está en memoria. Quien
    llama corta cuando le toca.

    Los desplazamientos se devuelven siempre al sistema de coordenadas del
    fragmento entero — sin eso, las menciones de la segunda mitad apuntarían a
    caracteres equivocados y la evidencia de las tripletas sería falsa.
    """
    if len(texto) <= ancho:
        return [(0, len(texto))] if texto.strip() else []

    paso = max(1, ancho - solape)
    tramos: list[tuple[int, int]] = []
    for inicio in range(0, len(texto), paso):
        fin = min(inicio + ancho, len(texto))
        if texto[inicio:fin].strip():
            tramos.append((inicio, fin))
        if fin >= len(texto):
            break
    return tramos


def _situar(texto: str, superficie: str, inicio, fin) -> tuple[int, int] | None:
    """Comprueba los offsets que declara el modelo contra la superficie que dice.

    Ninguna de las librerías de NER documenta si `start`/`end` son caracteres o
    tokens, y de esos dos números depende toda la evidencia de las tripletas: una
    convención distinta a la esperada no rompe nada visiblemente, solo llena el
    GraphML de citas que no dicen lo que la arista afirma. Se verifica el corte y,
    si no cuadra, se recurre a buscar la superficie; si tampoco aparece, la
    mención se descarta en vez de inventarle una posición.
    """
    if isinstance(inicio, int) and isinstance(fin, int) and texto[inicio:fin] == superficie:
        return inicio, fin
    hallado = texto.find(superficie)
    if hallado < 0:
        return None
    return hallado, hallado + len(superficie)


# --- Backends de NER ------------------------------------------------------


class NerGliner2:
    """`fastino/gliner2-multi-v1` sobre PyTorch, vía el paquete oficial `gliner2`.

    **Por qué este y no el export ONNX.** El mismo modelo y los mismos pesos
    (Apache-2.0, 205M), pero con las dos cosas que a `gliner2-onnx` le faltan y
    que aquí deciden el orden de magnitud: el modelo se coloca en la GPU
    (`map_location`) y expone `batch_extract_entities`, que infiere sobre una
    lista de textos de una vez. El binding Python de `gliner2-onnx` solo tiene
    `extract_entities(text, labels)`, una llamada por texto, y por defecto
    —`providers=None`— abre la sesión con `CPUExecutionProvider`: sobre el corpus
    entero son ~160.000 inferencias sueltas en CPU, que es de donde salían las 15
    horas medidas.

    **La ventana no cambia.** El texto se sigue recorriendo por ventanas
    solapadas de `VENTANA` caracteres, porque la del modelo es más corta que el
    fragmento; lo que cambia es que las ventanas de todos los fragmentos viajan
    juntas al modelo en vez de una a una. `reconocer_corpus` hace el aplanado.

    **Las dependencias no chocan.** `gliner2[local]` declara `torch` y
    `transformers` sin tope de versión, así que convive con el `>=5.14.1` del
    proyecto — el conflicto que dejó fuera a `gliner` v1 y a `optimum` era con
    aquellos, no con este.
    """

    def __init__(
        self,
        modelo: str = "fastino/gliner2-multi-v1",
        dispositivo: str = "cuda",
        lote: int = 32,
    ):
        from gliner2 import GLiNER2  # dependencia opcional

        self.modelo = GLiNER2.from_pretrained(modelo, map_location=dispositivo)
        # Dónde quedaron los parámetros de verdad, no dónde se pidió que quedaran.
        # Las 15 horas de la primera versión fueron una corrida en CPU que nada
        # delató: cada backend declara aquí su dispositivo y `04_grafo` lo imprime,
        # para que la propia corrida lo atestigüe en vez de deducirlo del reloj.
        import torch

        d = next(self.modelo.parameters()).device
        self.donde = (
            f"{d} ({torch.cuda.get_device_name(d)})" if d.type == "cuda" else str(d)
        )
        self.dispositivo = dispositivo
        self.lote = lote
        self.etiquetas = list(TIPOS.values())
        self.tipo_de = {v: k for k, v in TIPOS.items()}
        self.desalineadas = 0

    def procesar_lote(self, textos: list[str]) -> list[list[tuple[str, str, int, int, float]]]:
        crudos = self.modelo.batch_extract_entities(
            textos,
            self.etiquetas,
            batch_size=self.lote,
            # El umbral se le pasa al modelo en vez de filtrar después: lo que no
            # llega a `UMBRAL_CONFIANZA` no hace falta ni decodificarlo.
            threshold=UMBRAL_CONFIANZA,
            include_confidence=True,
            include_spans=True,
        )
        return [self._normalizar(t, c) for t, c in zip(textos, crudos)]

    def _normalizar(self, texto: str, crudo) -> list[tuple[str, str, int, int, float]]:
        """De la salida por etiqueta del modelo a la tupla que espera el módulo.

        `batch_extract_entities` devuelve un diccionario indexado por etiqueta, y
        según la versión lo envuelve o no en `{"entities": …}`. Se aceptan las dos
        formas: ninguna de las nueve etiquetas se llama «entities», así que la
        comprobación no puede confundirse.
        """
        if not isinstance(crudo, dict):
            return []
        anidado = crudo.get("entities")
        por_etiqueta = anidado if isinstance(anidado, dict) else crudo

        salida: list[tuple[str, str, int, int, float]] = []
        for etiqueta, encontrados in por_etiqueta.items():
            tipo = self.tipo_de.get(etiqueta)
            if tipo is None or not isinstance(encontrados, list):
                continue
            for e in encontrados:
                if not isinstance(e, dict):
                    continue
                superficie = e.get("text") or ""
                confianza = float(e.get("confidence", 1.0))
                if not superficie or confianza < UMBRAL_CONFIANZA:
                    continue
                tramo = _situar(texto, superficie, e.get("start"), e.get("end"))
                if tramo is None:
                    self.desalineadas += 1
                    continue
                salida.append((superficie, tipo, tramo[0], tramo[1], confianza))

        salida.sort(key=lambda t: (t[2], t[3]))
        return salida

    def __call__(self, texto: str) -> list[tuple[str, str, int, int, float]]:
        return self.procesar_lote([texto])[0]


class NerOnnx:
    """`fastino/gliner2-multi-v1` exportado a ONNX, vía `gliner2-onnx` (MIT).

    Se conserva por la Radeon: es el único camino a DirectML, que es la GPU que
    tiene la máquina de desarrollo. En NVIDIA hay que preferir `NerGliner2`, que
    sí infiere por lotes.

    **El proveedor hay que pasarlo.** Con `providers=None`, `gliner2-onnx` abre la
    sesión con `["CPUExecutionProvider"]` y no avisa: la GPU no se intenta usar,
    no es que falle y caiga a CPU. Por eso el defecto de aquí es explícito y por
    eso `04_grafo.py` lo expone en la línea de órdenes.
    """

    VENTANA = VENTANA
    SOLAPE = SOLAPE

    # `lmo3/` y no `lion-ai/`: el primero es el que publica y documenta el propio
    # paquete `gliner2-onnx`; el segundo es un espejo de un tercero, creado en
    # marzo de 2026 y sin descargas, que nadie ha comprobado que traiga los mismos
    # pesos.
    MODELO = "lmo3/gliner2-multi-v1-onnx"

    def __init__(
        self,
        modelo: str | None = None,
        proveedores: list[str] | None = None,
    ):
        from gliner2_onnx import GLiNER2ONNXRuntime  # dependencia opcional

        self.runtime = GLiNER2ONNXRuntime.from_pretrained(
            modelo or self.MODELO,
            providers=proveedores or ["CPUExecutionProvider"],
        )
        # Los proveedores con que la sesión abrió de verdad. ONNX Runtime descarta
        # sin error los que no pueda cargar —pedir DmlExecutionProvider sin la DLL
        # deja la sesión en CPU y nada lo dice—, así que se le pregunta a la sesión
        # y no a la petición.
        self.donde = ", ".join(self.runtime.encoder.get_providers())
        self.etiquetas = list(TIPOS.values())
        self.tipo_de = {v: k for k, v in TIPOS.items()}

    def procesar_lote(self, textos: list[str]) -> list[list[tuple[str, str, int, int, float]]]:
        """Secuencial a la fuerza: esta librería no expone inferencia por lotes.

        El binding JavaScript tiene `extractEntitiesBatch`; el de Python, no. El
        método existe para que este backend cumpla la misma interfaz que los
        otros, no porque aquí haya nada que ganar.
        """
        return [self(texto) for texto in textos]

    def __call__(self, texto: str) -> list[tuple[str, str, int, int, float]]:
        crudas: dict[tuple[int, int], tuple[str, str, int, int, float]] = {}
        for inicio_v, fin_v in ventanas(texto, self.VENTANA, self.SOLAPE):
            trozo = texto[inicio_v:fin_v]
            for e in self.runtime.extract_entities(trozo, self.etiquetas):
                tipo = self.tipo_de.get(e.label)
                if tipo is None or e.score < UMBRAL_CONFIANZA:
                    continue
                tramo = _situar(trozo, e.text, e.start, e.end)
                if tramo is None:
                    continue
                inicio, fin = tramo[0] + inicio_v, tramo[1] + inicio_v
                # El solape entre ventanas hace que una entidad de la zona común
                # se proponga dos veces. Se conserva la de mayor confianza, que es
                # normalmente la de la ventana donde queda más centrada.
                previa = crudas.get((inicio, fin))
                if previa is None or e.score > previa[4]:
                    crudas[(inicio, fin)] = (e.text, tipo, inicio, fin, float(e.score))
        return [crudas[k] for k in sorted(crudas)]


class NerFalso:
    """Backend determinista para las pruebas y para dibujar el grafo sin GPU.

    Reconoce las mayúsculas iniciales encadenadas, que es un NER pésimo y no
    pretende otra cosa: existe para que la construcción del grafo, la poda y la
    exportación a GraphML se puedan ejercitar en `pytest` sin descargar un
    modelo de 500 MB ni depender de la red.
    """

    PATRON = re.compile(r"\b[A-ZÁÉÍÓÚÑ][\w\-]+(?:\s+[A-ZÁÉÍÓÚÑ][\w\-]+)*")

    def procesar_lote(self, textos: list[str]) -> list[list[tuple[str, str, int, int, float]]]:
        return [self(texto) for texto in textos]

    def __call__(self, texto: str) -> list[tuple[str, str, int, int, float]]:
        salida = []
        for m in self.PATRON.finditer(texto):
            if admisible(m.group()):
                salida.append((m.group(), "organizacion", m.start(), m.end(), 1.0))
        return salida


def cargar_backend(
    nombre: str,
    modelo: str | None = None,
    dispositivo: str = "cuda",
    lote: int = 32,
    proveedores: list[str] | None = None,
):
    if nombre == "falso":
        return NerFalso()
    if nombre == "gliner2":
        opciones = {"dispositivo": dispositivo, "lote": lote}
        return NerGliner2(modelo, **opciones) if modelo else NerGliner2(**opciones)
    if nombre == "onnx":
        return NerOnnx(modelo, proveedores)
    raise ValueError(f"backend de NER desconocido: {nombre}")


# --- Extracción sobre el corpus -------------------------------------------


# Cuántos lotes del modelo se le entregan al backend en cada llamada. El lote lo
# microgestiona el propio `batch_extract_entities`; esto solo agrupa para que el
# bucle de Python no vuelva a ser el cuello de botella y para dar progreso con
# grano fino.
LOTES_POR_TANDA = 8


def reconocer_corpus(
    fragmentos: list[dict],
    backend,
    tam_lote: int = 32,
    progreso=None,
) -> dict[str, list[Mencion]]:
    """NER sobre el corpus entero, en lotes y con una sola pasada de ventaneo.

    **Es aquí donde se gana el tiempo.** La versión anterior recorría los
    fragmentos de uno en uno y llamaba al modelo una vez por ventana: ~160.000
    inferencias sueltas, cada una pagando su propio arranque. Aquí las ventanas de
    todos los fragmentos se aplanan en una sola cola y bajan al modelo en lotes,
    que es la forma en que una GPU rinde lo que dice su ficha.

    El ventaneo, la deduplicación del solape y el remapeo de offsets no dependen
    del backend, así que viven aquí y se pueden ejercitar con `NerFalso` en
    `pytest` —sin GPU y sin red— exactamente igual que corren con el modelo real.
    """
    cola: list[tuple[int, int, int]] = []  # (índice de fragmento, inicio, fin)
    for i, fragmento in enumerate(fragmentos):
        for inicio, fin in ventanas(fragmento["texto"]):
            cola.append((i, inicio, fin))

    # (índice de fragmento) -> (inicio, fin) -> mención cruda. El diccionario
    # interno resuelve el solape: una entidad de la zona común entre dos ventanas
    # se propone dos veces y se conserva la de mayor confianza.
    acumulado: dict[int, dict[tuple[int, int], tuple[str, str, int, int, float]]] = {}
    tanda = max(1, tam_lote) * LOTES_POR_TANDA

    for arranque in range(0, len(cola), tanda):
        piezas = cola[arranque : arranque + tanda]
        textos = [fragmentos[i]["texto"][inicio:fin] for i, inicio, fin in piezas]
        for (indice, desplazamiento, _), encontradas in zip(
            piezas, backend.procesar_lote(textos)
        ):
            destino = acumulado.setdefault(indice, {})
            for superficie, tipo, inicio, fin, confianza in encontradas:
                if not admisible(superficie):
                    continue
                tramo = (inicio + desplazamiento, fin + desplazamiento)
                previa = destino.get(tramo)
                if previa is None or confianza > previa[4]:
                    destino[tramo] = (superficie, tipo, tramo[0], tramo[1], confianza)
        if progreso:
            progreso(min(arranque + tanda, len(cola)), len(cola))

    menciones: dict[str, list[Mencion]] = {}
    for indice, hallazgos in acumulado.items():
        if not hallazgos:
            continue
        fragmento = fragmentos[indice]
        menciones[fragmento["chunk_id"]] = [
            Mencion(
                chunk_id=fragmento["chunk_id"],
                doc_id=fragmento["doc_id"],
                texto=hallazgos[tramo][0],
                tipo=hallazgos[tramo][1],
                inicio=tramo[0],
                fin=tramo[1],
                confianza=hallazgos[tramo][4],
            )
            for tramo in sorted(hallazgos)
        ]
    return menciones


def menciones_de_fragmento(fragmento: dict, backend) -> list[Mencion]:
    return [
        Mencion(
            chunk_id=fragmento["chunk_id"],
            doc_id=fragmento["doc_id"],
            texto=texto,
            tipo=tipo,
            inicio=inicio,
            fin=fin,
            confianza=confianza,
        )
        for texto, tipo, inicio, fin, confianza in backend(fragmento["texto"])
        if admisible(texto)
    ]


def escribir_cache(menciones_por_chunk: dict[str, list[Mencion]], destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for chunk_id in sorted(menciones_por_chunk):
            menciones = menciones_por_chunk[chunk_id]
            fh.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": menciones[0].doc_id if menciones else "",
                        "menciones": [
                            {
                                "texto": m.texto,
                                "tipo": m.tipo,
                                "inicio": m.inicio,
                                "fin": m.fin,
                                "confianza": round(m.confianza, 4),
                            }
                            for m in menciones
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return destino


def leer_cache(ruta: Path) -> dict[str, list[Mencion]]:
    salida: dict[str, list[Mencion]] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            salida[registro["chunk_id"]] = [
                Mencion(
                    chunk_id=registro["chunk_id"],
                    doc_id=registro["doc_id"],
                    texto=m["texto"],
                    tipo=m["tipo"],
                    inicio=m["inicio"],
                    fin=m["fin"],
                    confianza=m["confianza"],
                )
                for m in registro["menciones"]
            ]
    return salida


# --- Canonicalización -----------------------------------------------------


@dataclass
class Entidad:
    """Un nodo del grafo: todas las formas de nombrar una misma cosa."""

    id: str  # la clave normalizada de la forma más frecuente
    nombre: str  # la forma superficial más frecuente, para leerlo
    tipo: str  # el tipo mayoritario entre sus menciones
    alias: tuple[str, ...]  # claves normalizadas que colapsan aquí
    frecuencia_documental: int  # en cuántos documentos distintos aparece


def canonicalizar(
    menciones: list[Mencion],
    equivalencias: dict[str, str] | None = None,
) -> tuple[dict[str, Entidad], dict[str, str]]:
    """Agrupa las menciones en entidades. Devuelve (entidades, clave -> id).

    El agrupamiento base es por clave normalizada, que resuelve mayúsculas,
    tildes, puntuación y artículos. Lo que **no** resuelve es el cruce de idiomas:
    «Estados Unidos», «United States» y «Estados Unidos da América» siguen siendo
    tres nodos, y en un corpus trilingüe eso parte el grafo justo por donde más
    caminos habría.

    `equivalencias` es el mapa que cierra ese hueco (alias -> id canónico) y lo
    produce `agrupar_por_embedding`, que codifica los nombres con el mismo encoder
    que ya construye el índice. Se pasa como argumento en vez de calcularse aquí
    para que esta función siga siendo pura y comprobable sin cargar un modelo.
    """
    equivalencias = equivalencias or {}
    por_id: dict[str, dict] = {}

    for mencion in menciones:
        clave = mencion.clave
        if not clave:
            continue
        destino = equivalencias.get(clave, clave)
        acumulado = por_id.setdefault(
            destino,
            {"formas": {}, "tipos": {}, "docs": set(), "alias": set()},
        )
        acumulado["formas"][mencion.texto] = acumulado["formas"].get(mencion.texto, 0) + 1
        acumulado["tipos"][mencion.tipo] = acumulado["tipos"].get(mencion.tipo, 0) + 1
        acumulado["docs"].add(mencion.doc_id)
        acumulado["alias"].add(clave)

    entidades: dict[str, Entidad] = {}
    clave_a_id: dict[str, str] = {}
    for id_entidad, acumulado in por_id.items():
        # Desempates por frecuencia y luego alfabéticos: sin el segundo criterio,
        # dos formas igual de frecuentes darían nombres distintos según el orden
        # de iteración y el grafo dejaría de ser reproducible.
        nombre = max(sorted(acumulado["formas"]), key=lambda f: acumulado["formas"][f])
        tipo = max(sorted(acumulado["tipos"]), key=lambda t: acumulado["tipos"][t])
        entidades[id_entidad] = Entidad(
            id=id_entidad,
            nombre=nombre,
            tipo=tipo,
            alias=tuple(sorted(acumulado["alias"])),
            frecuencia_documental=len(acumulado["docs"]),
        )
        for alias in acumulado["alias"]:
            clave_a_id[alias] = id_entidad

    return entidades, clave_a_id


# Cuántos vecinos se examinan por nombre al agrupar. Una entidad tiene unas pocas
# formas de citarse —el nombre, la sigla y su traducción a los otros dos idiomas
# del corpus— así que ocho candidatos cubren de sobra el grupo real; lo que quede
# por debajo del octavo vecino no iba a pasar un umbral de 0,92.
VECINOS_AGRUPAMIENTO = 8


def agrupar_por_embedding(
    claves: list[str],
    codificar,
    umbral: float = 0.92,
    vecinos: int = VECINOS_AGRUPAMIENTO,
) -> dict[str, str]:
    """Une nombres de entidad que significan lo mismo en idiomas distintos.

    `codificar` recibe una lista de textos y devuelve una matriz de vectores
    normalizados: en la práctica, el encoder que ya construyó el índice. Reusarlo
    tiene tres ventajas sobre traer un modelo de *entity linking*: es cross-lingüe
    por construcción —para eso se eligió—, ya está cargado en memoria durante la
    etapa, y no añade ninguna licencia nueva que declarar.

    **Se busca por vecinos y no por matriz completa.** La versión obvia,
    `vectores @ vectores.T`, es cuadrática: sobre los ~70.000 nombres distintos
    que salen del corpus son 19 GB de similitudes para quedarse con un puñado de
    pares. Con FAISS —que ya es dependencia del proyecto y hace exactamente esta
    búsqueda— el coste baja a los `vecinos` más próximos de cada nombre, que es
    todo lo que el umbral puede aceptar.

    El umbral es alto a propósito. Con nombres cortos y sin contexto, el coseno
    entre dos entidades del mismo dominio sube mucho —«Fuerza Aérea Colombiana» y
    «Fuerza Aérea Brasileña» se parecen— así que agrupar de más funde nodos que
    deberían competir. **Sin calibrar todavía sobre este corpus**: 0,92 es un
    punto de partida conservador, y la forma de fijarlo es anotar un centenar de
    pares y mirar dónde se cruzan las curvas, como se hizo con el umbral de
    `emparejamiento`.
    """
    import faiss
    import numpy as np

    if len(claves) < 2:
        return {}

    vectores = np.ascontiguousarray(codificar(claves), dtype=np.float32)
    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)
    # +1 porque el primer vecino de un vector es siempre él mismo.
    similitudes, posiciones = indice.search(vectores, min(vecinos + 1, len(claves)))

    # Unión de vecinos con compresión de caminos. El representante de cada grupo
    # es siempre el de menor índice, y el orden de `claves` lo fija quien llama
    # (ordenado alfabéticamente), así que el agrupamiento no depende del hardware
    # ni del orden en que FAISS devuelva los empates.
    padre = list(range(len(claves)))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(len(claves)):
        for similitud, j in zip(similitudes[i], posiciones[i]):
            if j < 0 or j == i or similitud < umbral:
                continue
            ri, rj = raiz(i), raiz(int(j))
            if ri != rj:
                padre[max(ri, rj)] = min(ri, rj)

    equivalencias: dict[str, str] = {}
    for i, clave in enumerate(claves):
        canonica = claves[raiz(i)]
        if canonica != clave:
            equivalencias[clave] = canonica
    return equivalencias
