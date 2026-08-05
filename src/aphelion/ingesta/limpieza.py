"""Normalización del texto extraído.

Dos operaciones no triviales:

1. **Boilerplate por frecuencia.** En vez de patrones fijos ("Página N de M"), se
   detectan las líneas cortas que se repiten a lo largo del documento. Un
   encabezado institucional aparece en cada página; una frase del cuerpo no. El
   criterio es estructural y funciona igual en los tres idiomas del corpus.

2. **Idioma predominante.** Se estima con palabras funcionales, que son las de
   mayor frecuencia y menor ambigüedad entre español, inglés y portugués. Basta
   para post-filtros de metadata y evita una dependencia adicional.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Caracteres de control salvo tabulador y salto de línea.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ESPACIOS = re.compile(r"[ \t  - ]+")
_SALTOS = re.compile(r"\n{3,}")
# Ligaduras tipográficas que PyMuPDF conserva y rompen la coincidencia léxica.
_LIGADURAS = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

_SOLO_NUMERO = re.compile(r"^[\s\d\W]{0,12}$")

_FUNCIONALES = {
    "es": {"de", "la", "que", "el", "en", "los", "del", "las", "por", "para", "con", "una", "es"},
    "en": {"the", "of", "and", "to", "in", "for", "is", "that", "with", "on", "as", "by", "are"},
    "pt": {"de", "da", "do", "que", "em", "para", "com", "uma", "os", "as", "no", "na", "não"},
}


def normalizar(texto: str) -> str:
    """Unicode NFC, sin caracteres de control ni espacios redundantes."""
    if not texto:
        return ""
    for ligadura, reemplazo in _LIGADURAS.items():
        texto = texto.replace(ligadura, reemplazo)
    texto = unicodedata.normalize("NFC", texto)
    texto = _CONTROL.sub(" ", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = _ESPACIOS.sub(" ", texto)
    texto = "\n".join(linea.strip() for linea in texto.split("\n"))
    return _SALTOS.sub("\n\n", texto).strip()


def lineas_boilerplate(
    texto: str, min_repeticiones: int = 4, max_long: int = 90
) -> set[str]:
    """Qué líneas descartaría `quitar_boilerplate`, sin descartarlas todavía.

    Está separado porque la decisión es **del documento entero** —una cabecera se
    reconoce por repetirse entre páginas— pero se aplica por tramos cuando se
    fragmenta por secciones. Calculando el conjunto una vez y filtrando después,
    las dos estrategias de chunking reciben exactamente el mismo texto limpio, y
    lo que las separa es dónde cortan y no qué preprocesado les tocó.
    """
    lineas = texto.split("\n")
    if len(lineas) < min_repeticiones * 2:
        return set()

    frecuencias = Counter(l.strip() for l in lineas if l.strip())
    return {
        linea
        for linea, n in frecuencias.items()
        if n >= min_repeticiones and len(linea) <= max_long
    }


def quitar_boilerplate(texto: str, min_repeticiones: int = 4, max_long: int = 90) -> str:
    """Elimina líneas cortas que se repiten muchas veces en el mismo documento.

    `min_repeticiones` evita borrar frases legítimamente repetidas dos o tres
    veces; `max_long` evita borrar párrafos completos.
    """
    basura = lineas_boilerplate(texto, min_repeticiones, max_long)
    if not basura:
        return texto

    limpias = [l for l in texto.split("\n") if l.strip() not in basura]
    return _SALTOS.sub("\n\n", "\n".join(limpias)).strip()


def limpiar_seccion(texto: str, basura: set[str]) -> str:
    """La limpieza de `limpiar`, con el boilerplate ya decidido fuera.

    Mismo orden de operaciones que el pipeline completo; lo único que cambia es
    que el conjunto de líneas repetidas viene dado, porque una sección suelta no
    tiene repeticiones suficientes para reconocerlo por su cuenta.
    """
    texto = normalizar(texto)
    if basura:
        texto = "\n".join(l for l in texto.split("\n") if l.strip() not in basura)
    texto = quitar_numeracion(texto)
    return _SALTOS.sub("\n\n", texto).strip()


def quitar_numeracion(texto: str) -> str:
    """Descarta líneas que solo contienen números o signos (folios sueltos)."""
    return "\n".join(l for l in texto.split("\n") if not _SOLO_NUMERO.match(l) or not l.strip())


def detectar_idioma(texto: str) -> str:
    """Devuelve 'es', 'en', 'pt' o 'xx' si no hay señal suficiente."""
    palabras = re.findall(r"\b\w+\b", texto[:20000].lower())
    if len(palabras) < 20:
        return "xx"

    vistas = Counter(palabras)
    puntajes = {
        idioma: sum(vistas[p] for p in marcadores)
        for idioma, marcadores in _FUNCIONALES.items()
    }
    mejor = max(puntajes, key=puntajes.get)
    if puntajes[mejor] == 0:
        return "xx"

    # Español y portugués comparten muchas funcionales; se desempata con marcas
    # que solo existen en uno de los dos.
    if mejor in ("es", "pt"):
        pt_exclusivas = vistas["não"] + vistas["são"] + vistas["está"] + vistas["dos"]
        es_exclusivas = vistas["y"] + vistas["el"] + vistas["su"] + vistas["ha"]
        mejor = "pt" if pt_exclusivas > es_exclusivas else "es"

    return mejor


def limpiar(texto: str) -> tuple[str, str]:
    """Pipeline completo. Devuelve (texto_limpio, idioma)."""
    texto = normalizar(texto)
    texto = quitar_boilerplate(texto)
    texto = quitar_numeracion(texto)
    texto = _SALTOS.sub("\n\n", texto).strip()
    return texto, detectar_idioma(texto)
