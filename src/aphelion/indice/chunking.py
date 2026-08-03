"""Fragmentación con completitud lingüística garantizada.

Ninguna oración puede cruzar la frontera entre dos fragmentos. Eso descarta cortar por número de tokens y obliga a construir los
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
from tokenizers import Tokenizer

from .. import config

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
def _tokenizador(nombre_encoder: str) -> Tokenizer:
    """Carga el tokenizador por el backend Rust, no por `transformers`.

    Los ids son los mismos —ambos leen el `tokenizer.json` del encoder—, pero
    `transformers` importa torch de forma transitiva y eso son ~1,35 GiB de
    memoria comprometida por proceso. Fragmentar reparte el trabajo entre un
    proceso por núcleo, así que ese peso se multiplica por el número de workers
    y agota la memoria comprometible antes de leer el primer documento: el
    cargador de DLL de Windows aborta con `WinError 1455`. Por aquí son ~24 MiB.
    """
    return Tokenizer.from_pretrained(config.ENCODERS[nombre_encoder]["modelo"])


def _contar(tokenizador: Tokenizer, texto: str) -> int:
    return len(tokenizador.encode(texto, add_special_tokens=False).ids)


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


def _recortar_tabular(texto: str, max_tokens: int) -> str:
    """Descarta las filas que el tope tabular iba a tirar de todos modos.

    Sin esto el recorte llega tarde: `pysbd` segmenta los 47,8 MB del mayor CSV
    del corpus —321 s— para que después se descarten 37.951 de los 38.351
    fragmentos. Recortar antes de segmentar convierte esa etapa en segundos.

    El corte cae en frontera de párrafo, que en los formatos tabulares es la
    frontera entre filas: nunca parte un registro por la mitad.
    """
    # 4 caracteres por token es la proporción típica en este corpus; el factor
    # 1.5 cubre la variación y el filtro de fragmentos cortos posterior.
    presupuesto = int(config.MAX_FRAGMENTOS_TABULARES * max_tokens * 4 * 1.5)
    if len(texto) <= presupuesto:
        return texto

    corte = texto.rfind("\n\n", 0, presupuesto)
    return texto[: corte if corte > 0 else presupuesto]


def _partir_por_tokens(texto: str, tokenizador: Tokenizer, presupuesto: int) -> list[str]:
    """Corta un texto en ventanas de `presupuesto` tokens usando offsets.

    Se tokeniza una sola vez y se cortan ventanas sobre el mapa de offsets. La
    alternativa evidente —ir añadiendo palabras y recontar— es cuadrática, y el
    corpus contiene bloques de más de 140.000 tokens (mapas PBF, filas de CSV
    muy anchas) donde esa diferencia decide si el proceso termina o no.
    """
    offsets = tokenizador.encode(texto, add_special_tokens=False).offsets
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


def _partir_oracion_larga(oracion: str, tokenizador: Tokenizer, presupuesto: int) -> list[str]:
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
    tokenizador: Tokenizer,
    max_tokens: int = config.CHUNK_TOKENS,
    solape: float = config.CHUNK_SOLAPE,
    max_fragmentos: int | None = None,
) -> list[tuple[str, int]]:
    """Agrupa oraciones en fragmentos. Devuelve (texto, num_tokens).

    El solape se aplica arrastrando las últimas oraciones del fragmento anterior,
    de modo que la frontera sigue cayendo en límites oracionales.

    `max_fragmentos` corta en cuanto se alcanza el tope. Importa: el CSV mayor
    del corpus produce 38.351 fragmentos y tokenizarlos todos para quedarse con
    400 desperdicia la mayor parte del tiempo de esta etapa.
    """
    if not oraciones:
        return []

    presupuesto_solape = int(max_tokens * solape)
    fragmentos: list[tuple[str, int]] = []
    actual: list[str] = []
    tokens_actual = 0

    for oracion in oraciones:
        if max_fragmentos is not None and len(fragmentos) >= max_fragmentos:
            actual = []  # lo acumulado se descarta: ya sobra
            break

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
    fragmentos: list[tuple[str, int]], tokenizador: Tokenizer, max_tokens: int
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

    formato = doc.get("formato") or doc["fuente"].rsplit(".", 1)[-1].lower()
    tabular = formato in config.FORMATOS_TABULARES

    if tabular:
        texto = _recortar_tabular(texto, max_tokens)

    tokenizador = _tokenizador(nombre_encoder)
    oraciones = dividir_en_oraciones(texto, idioma)
    grupos = agrupar(
        oraciones,
        tokenizador,
        max_tokens=max_tokens,
        # Se pide un margen sobre el tope porque el filtro de fragmentos cortos
        # que sigue puede descartar algunos.
        max_fragmentos=int(config.MAX_FRAGMENTOS_TABULARES * 1.2) if tabular else None,
    )

    # Fragmentos de una o dos palabras son restos de tablas y encabezados
    # sueltos: no aportan significado recuperable y solo compiten por espacio en
    # el ranking.
    grupos = [(t, n) for t, n in grupos if n >= config.MIN_TOKENS_FRAGMENTO]

    if tabular:
        grupos = grupos[: config.MAX_FRAGMENTOS_TABULARES]

    return [
        Fragmento(
            doc_id=doc["doc_id"],
            chunk_id=f"{doc['doc_id']}-chunk-{i:04d}",
            fuente=doc["fuente"],
            formato=formato,
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
