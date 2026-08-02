"""Fragmentación con completitud lingüística garantizada.

La especificación (§3.3) prohíbe que una oración cruce la frontera entre dos
fragmentos. Eso descarta cortar por número de tokens y obliga a construir los
fragmentos acumulando oraciones completas: cuando la siguiente no cabe dentro del
presupuesto, el fragmento se cierra donde terminó la anterior.

El conteo de tokens usa el tokenizador del encoder, no una aproximación por
palabras, porque el presupuesto que importa es el del modelo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pysbd

from . import config

# pysbd trae reglas por idioma; para el resto se usa el segmentador inglés, que
# es el más conservador ante abreviaturas desconocidas.
_IDIOMAS_PYSBD = {"es", "en", "pt"}


@dataclass(frozen=True)
class Fragmento:
    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int
    posicion: int
    num_tokens: int
    texto: str
    idioma: str
    observatorio: str
    ruta: str
    titulo: str | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "fuente": self.fuente,
            "formato": self.formato,
            "fenomeno": self.fenomeno,
            "posicion": self.posicion,
            "num_tokens": self.num_tokens,
            "texto": self.texto,
            "idioma": self.idioma,
            "observatorio": self.observatorio,
            "ruta": self.ruta,
            "titulo": self.titulo,
        }


@lru_cache(maxsize=8)
def _segmentador(idioma: str) -> pysbd.Segmenter:
    lang = idioma if idioma in _IDIOMAS_PYSBD else "en"
    return pysbd.Segmenter(language=lang, clean=False)


@lru_cache(maxsize=4)
def _tokenizador(nombre_encoder: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.ENCODERS[nombre_encoder]["modelo"])


def _contar(tokenizador, texto: str) -> int:
    return len(tokenizador.encode(texto, add_special_tokens=False))


def dividir_en_oraciones(texto: str, idioma: str) -> list[str]:
    """Segmenta respetando párrafos: nunca une texto de párrafos distintos.

    Importa para los formatos donde cada bloque es una unidad semántica propia
    (filas de CSV, registros de catálogo, elementos de mapa).
    """
    oraciones: list[str] = []
    for parrafo in texto.split("\n\n"):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        try:
            trozos = _segmentador(idioma).segment(parrafo)
        except Exception:
            trozos = [parrafo]
        oraciones.extend(t.strip() for t in trozos if t and t.strip())
    return oraciones


def _partir_por_tokens(texto: str, tokenizador, presupuesto: int) -> list[str]:
    """Corta un texto en ventanas de `presupuesto` tokens usando offsets.

    Se tokeniza una sola vez y se cortan ventanas sobre el mapa de offsets. La
    alternativa evidente —ir añadiendo palabras y recontar— es cuadrática, y el
    corpus contiene bloques de más de 140.000 tokens (mapas PBF, filas de CSV
    muy anchas) donde esa diferencia decide si el proceso termina o no.
    """
    codificado = tokenizador(
        texto, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = codificado["offset_mapping"]
    if not offsets:
        return []

    # Margen de seguridad: el texto recortado por offsets no siempre vuelve a
    # tokenizarse en la misma cantidad de piezas — la frontera puede partir un
    # subtoken y sumar uno. Sin este margen algunos trozos salen a 513.
    ventana_util = max(1, presupuesto - 2)

    trozos: list[str] = []
    for inicio in range(0, len(offsets), ventana_util):
        ventana = offsets[inicio : inicio + ventana_util]
        desde = ventana[0][0]
        hasta = ventana[-1][1]
        trozo = texto[desde:hasta].strip()
        if trozo:
            trozos.append(trozo)
    return trozos


def _partir_oracion_larga(oracion: str, tokenizador, presupuesto: int) -> list[str]:
    """Último recurso: una sola oración que excede el presupuesto del modelo.

    Ocurre con tablas volcadas a texto y con filas de CSV muy anchas, que no
    tienen puntuación interna. Se intenta primero cortar por separadores débiles
    (`;`, `,`, `|`), que preservan unidades legibles; lo que aún no quepa se
    trocea por ventanas de tokens. Es la única situación en la que el corte no
    cae en frontera oracional, y sucede porque descartar el contenido sería peor.
    """
    partes = re.split(r"(?<=[;,|])\s+", oracion)

    # Longitudes de una sola pasada: recontar por acumulación sería cuadrático.
    longitudes = [_contar(tokenizador, p) for p in partes]

    trozos: list[str] = []
    actual: list[str] = []
    tokens_actual = 0

    for parte, n in zip(partes, longitudes):
        if n > presupuesto:
            if actual:
                trozos.append(" ".join(actual))
                actual, tokens_actual = [], 0
            trozos.extend(_partir_por_tokens(parte, tokenizador, presupuesto))
            continue

        if actual and tokens_actual + n > presupuesto:
            trozos.append(" ".join(actual))
            actual, tokens_actual = [], 0

        actual.append(parte)
        tokens_actual += n

    if actual:
        trozos.append(" ".join(actual))

    return [t for t in trozos if t.strip()]


def agrupar(
    oraciones: list[str],
    tokenizador,
    max_tokens: int = config.CHUNK_TOKENS,
    solape: float = config.CHUNK_SOLAPE,
) -> list[tuple[str, int]]:
    """Agrupa oraciones en fragmentos. Devuelve (texto, num_tokens).

    El solape se aplica arrastrando las últimas oraciones del fragmento anterior,
    de modo que la frontera sigue cayendo en límites oracionales.
    """
    if not oraciones:
        return []

    presupuesto_solape = int(max_tokens * solape)
    fragmentos: list[tuple[str, int]] = []
    actual: list[str] = []
    tokens_actual = 0

    for oracion in oraciones:
        n = _contar(tokenizador, oracion)

        if n > max_tokens:
            if actual:
                fragmentos.append((" ".join(actual), tokens_actual))
                actual, tokens_actual = [], 0
            for trozo in _partir_oracion_larga(oracion, tokenizador, max_tokens):
                fragmentos.append((trozo, _contar(tokenizador, trozo)))
            continue

        if actual and tokens_actual + n > max_tokens:
            fragmentos.append((" ".join(actual), tokens_actual))
            # Arrastre para el solape: las últimas oraciones completas que quepan.
            arrastre: list[str] = []
            acumulado = 0
            for previa in reversed(actual):
                t = _contar(tokenizador, previa)
                if acumulado + t > presupuesto_solape:
                    break
                arrastre.insert(0, previa)
                acumulado += t
            actual, tokens_actual = arrastre, acumulado

        actual.append(oracion)
        tokens_actual += n

    if actual:
        fragmentos.append((" ".join(actual), tokens_actual))

    return _verificar_presupuesto(fragmentos, tokenizador, max_tokens)


def _verificar_presupuesto(
    fragmentos: list[tuple[str, int]], tokenizador, max_tokens: int
) -> list[tuple[str, int]]:
    """Reconta los fragmentos que rozan el límite y parte los que lo exceden.

    Sumar los tokens de cada oración no equivale a tokenizar el texto ya unido:
    al concatenar, el tokenizador puede fusionar o dividir piezas en las uniones.
    La desviación es pequeña pero real, y basta para que algún fragmento supere el
    presupuesto — lo que provocaría truncamiento silencioso en encoders de
    ventana corta como mE5-large.

    Solo se recuenta lo que está por encima del 90% del presupuesto, de modo que
    la verificación no duplica el coste de tokenización del corpus completo.
    """
    umbral = int(max_tokens * 0.9)
    verificados: list[tuple[str, int]] = []

    for texto, estimado in fragmentos:
        if estimado < umbral:
            verificados.append((texto, estimado))
            continue

        real = _contar(tokenizador, texto)
        if real <= max_tokens:
            verificados.append((texto, real))
            continue

        for trozo in _partir_por_tokens(texto, tokenizador, max_tokens):
            verificados.append((trozo, _contar(tokenizador, trozo)))

    return verificados


def fragmentar(
    doc: dict,
    texto: str,
    idioma: str,
    nombre_encoder: str = config.ENCODER_PRINCIPAL,
    max_tokens: int = config.CHUNK_TOKENS,
) -> list[Fragmento]:
    """Convierte un documento limpio en su lista de fragmentos."""
    if not texto.strip():
        return []

    tokenizador = _tokenizador(nombre_encoder)
    oraciones = dividir_en_oraciones(texto, idioma)
    grupos = agrupar(oraciones, tokenizador, max_tokens=max_tokens)

    # Fragmentos de una o dos palabras son restos de tablas y encabezados
    # sueltos: no aportan significado recuperable y solo compiten por espacio en
    # el ranking.
    grupos = [(t, n) for t, n in grupos if n >= config.MIN_TOKENS_FRAGMENTO]

    return [
        Fragmento(
            doc_id=doc["doc_id"],
            chunk_id=f"{doc['doc_id']}-chunk-{i:04d}",
            fuente=doc["fuente"],
            formato=doc.get("formato") or doc["fuente"].rsplit(".", 1)[-1].lower(),
            fenomeno=doc["fenomeno"],
            posicion=i,
            num_tokens=n,
            texto=t,
            idioma=idioma,
            observatorio=doc.get("observatorio", ""),
            ruta=doc.get("ruta_rel", ""),
            titulo=(doc.get("meta") or {}).get("titulo"),
        )
        for i, (t, n) in enumerate(grupos)
    ]
