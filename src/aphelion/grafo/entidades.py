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

**Por qué el backend por defecto es ONNX.** Dos razones que se suman: el
proyecto ya monta DirectML para la Radeon (`indice/onnx_dml.py`), y el paquete
`gliner` v1 declara `transformers<5.14.0` mientras el proyecto tiene fijado
`>=5.14.1` — un conflicto que `uv sync` rechaza, el mismo que ya dejó fuera a
`optimum`. `gliner2-onnx` pide `transformers>=4.40` sin tope y no arrastra torch.
Si aun así se prefiere el backend `gliner`, se ejecuta esta etapa aislada:

    uv run --isolated --with gliner2-onnx python scripts/etapas/04_grafo.py

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


# --- Backends de NER ------------------------------------------------------


class NerOnnx:
    """`fastino/gliner2-multi-v1` exportado a ONNX, vía `gliner2-onnx` (MIT).

    El proveedor de ejecución se elige igual que en `indice/onnx_dml.py`: DirectML
    en la Radeon, CPU en cualquier otro sitio. La ventana del modelo es corta
    (384 tokens frente a los 504 del fragmento), así que el texto se recorre por
    ventanas solapadas y los desplazamientos se devuelven al sistema de
    coordenadas del fragmento entero — sin eso, las menciones de la segunda mitad
    apuntarían a caracteres equivocados y la evidencia de las tripletas sería
    falsa.
    """

    # En caracteres, no en tokens: aquí no hay tokenizador y no hace falta, porque
    # lo que se busca es quedarse holgadamente por debajo de la ventana. 1200
    # caracteres son ~300 tokens en inglés y menos en español.
    VENTANA = 1200
    SOLAPE = 200

    def __init__(self, modelo: str = "lion-ai/gliner2-multi-v1-onnx", proveedores=None):
        from gliner2_onnx import GLiNER2ONNXRuntime  # dependencia opcional

        self.runtime = GLiNER2ONNXRuntime.from_pretrained(
            modelo, providers=proveedores
        ) if proveedores else GLiNER2ONNXRuntime.from_pretrained(modelo)
        self.etiquetas = list(TIPOS.values())
        self.tipo_de = {v: k for k, v in TIPOS.items()}

    def _ventanas(self, texto: str):
        if len(texto) <= self.VENTANA:
            yield 0, texto
            return
        paso = self.VENTANA - self.SOLAPE
        for inicio in range(0, len(texto), paso):
            trozo = texto[inicio : inicio + self.VENTANA]
            if trozo.strip():
                yield inicio, trozo
            if inicio + self.VENTANA >= len(texto):
                break

    def __call__(self, texto: str) -> list[tuple[str, str, int, int, float]]:
        crudas: dict[tuple[int, int], tuple[str, str, int, int, float]] = {}
        for desplazamiento, trozo in self._ventanas(texto):
            for e in self.runtime.extract_entities(trozo, self.etiquetas):
                tipo = self.tipo_de.get(e.label)
                if tipo is None or e.score < UMBRAL_CONFIANZA:
                    continue
                inicio, fin = e.start + desplazamiento, e.end + desplazamiento
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

    def __call__(self, texto: str) -> list[tuple[str, str, int, int, float]]:
        salida = []
        for m in self.PATRON.finditer(texto):
            if admisible(m.group()):
                salida.append((m.group(), "organizacion", m.start(), m.end(), 1.0))
        return salida


def cargar_backend(nombre: str, modelo: str | None = None):
    if nombre == "falso":
        return NerFalso()
    if nombre == "onnx":
        return NerOnnx(modelo) if modelo else NerOnnx()
    raise ValueError(f"backend de NER desconocido: {nombre}")


# --- Extracción sobre el corpus -------------------------------------------


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
