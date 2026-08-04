"""Extracción de texto por formato.

Cada extractor devuelve un `Extraccion`: el texto plano del documento más los
campos descriptivos que convenga conservar como metadata en lugar de mezclar con
el cuerpo, que es lo recomendado para los JSON de artículo.

Los PDF sin capa de texto se detectan aquí y se marcan con `necesita_ocr`; su
procesamiento se resuelve en `ocr.py`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import config
from .catalogo import Documento


@dataclass
class Extraccion:
    texto: str
    meta: dict[str, Any] = field(default_factory=dict)
    necesita_ocr: bool = False
    error: str | None = None

    @property
    def vacia(self) -> bool:
        return not self.texto.strip()


# --- PDF ------------------------------------------------------------------


def _extraer_pdf(ruta: Path) -> Extraccion:
    import pymupdf

    try:
        doc = pymupdf.open(ruta)
    except Exception as e:
        return Extraccion("", error=f"pdf ilegible: {e}")

    partes: list[str] = []
    with doc:
        meta = {
            "titulo": (doc.metadata or {}).get("title") or None,
            "paginas": doc.page_count,
        }
        for pagina in doc:
            # sort=True reordena los bloques según el flujo de lectura, lo que
            # importa en los documentos a dos columnas (atlas RESDAL, informes CSIS).
            partes.append(pagina.get_text("text", sort=True))

    texto = "\n\n".join(partes)
    necesita_ocr = len(texto.strip()) < config.UMBRAL_PDF_SIN_TEXTO
    return Extraccion(texto, meta, necesita_ocr=necesita_ocr)


# --- JSON -----------------------------------------------------------------

# Claves que contienen el cuerpo del texto, por orden de preferencia.
#
# `abstract` va al final y no entre las de metadata: es lo único sustantivo que
# traen los 80 artículos de la revista del CEEEP, que no tienen cuerpo. Sin él
# esos documentos entraban al índice con solo su título —15 a 39 tokens— y
# perdían 88.975 caracteres de resumen en español sobre seguridad y defensa,
# justo el material que responde varias consultas de los fenómenos 1 y 3. Un
# documento con cuerpo propio no se ve afectado: la primera clave que aparece
# es la que manda.
_CLAVES_CUERPO = ("body_paragraphs", "body_text", "sections", "content", "texto", "abstract")
# Claves descriptivas: van a metadata, no al cuerpo.
_CLAVES_META = ("url", "date", "fecha", "authors", "tags", "topics", "excerpt")


def _aplanar(valor: Any) -> str:
    """Convierte cualquier estructura anidada en texto legible."""
    if valor is None:
        return ""
    if isinstance(valor, str):
        return valor.strip()
    if isinstance(valor, (int, float, bool)):
        return str(valor)
    if isinstance(valor, list):
        return "\n".join(p for p in (_aplanar(v) for v in valor) if p)
    if isinstance(valor, dict):
        partes = []
        for k, v in valor.items():
            texto = _aplanar(v)
            if texto:
                partes.append(f"{k}: {texto}" if len(texto) < 200 else texto)
        return "\n".join(partes)
    return str(valor)


def _extraer_json_articulo(d: dict) -> Extraccion:
    partes: list[str] = []

    titulo = d.get("title") or d.get("titulo")
    if titulo:
        partes.append(str(titulo).strip())

    for clave in _CLAVES_CUERPO:
        if clave in d:
            cuerpo = _aplanar(d[clave])
            if cuerpo:
                partes.append(cuerpo)
                break

    # Las alertas de la Defensoría guardan lo sustantivo aquí: tema, municipios,
    # grupos armados. Sin esto quedarían reducidas a un párrafo suelto.
    for clave in ("alerta_meta", "fields", "lists"):
        if clave in d:
            extra = _aplanar(d[clave])
            if extra:
                partes.append(extra)

    meta = {k: d[k] for k in _CLAVES_META if k in d and d[k]}
    if titulo:
        meta["titulo"] = str(titulo).strip()

    return Extraccion("\n\n".join(p for p in partes if p), meta)


def _extraer_json_catalogo(registros: list) -> Extraccion:
    """Los catálogos son listas de registros que describen otros archivos.

    Cada registro se emite como un bloque separado; el chunking posterior los
    respeta como unidades independientes.
    """
    bloques = [b for b in (_aplanar(r) for r in registros) if b]
    return Extraccion(
        "\n\n".join(bloques),
        {"registros": len(registros), "es_catalogo": True},
    )


def _extraer_json(ruta: Path) -> Extraccion:
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception as e:
        return Extraccion("", error=f"json ilegible: {e}")

    if isinstance(datos, list):
        return _extraer_json_catalogo(datos)
    if isinstance(datos, dict):
        return _extraer_json_articulo(datos)
    return Extraccion(_aplanar(datos))


# --- Tabulares ------------------------------------------------------------


def _fila_a_texto(cabecera: list[str], valores: list) -> str:
    """`columna: valor` por celda no vacía, para que el valor conserve contexto."""
    pares = [
        f"{col}: {val}"
        for col, val in zip(cabecera, valores)
        if val is not None and str(val).strip()
    ]
    return " | ".join(pares)


def _extraer_csv(ruta: Path) -> Extraccion:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with ruta.open(encoding=encoding, newline="") as fh:
                filas = list(csv.reader(fh))
            break
        except (UnicodeDecodeError, csv.Error):
            continue
    else:
        return Extraccion("", error="csv ilegible")

    if not filas:
        return Extraccion("")

    cabecera = [str(c) for c in filas[0]]
    bloques = [_fila_a_texto(cabecera, f) for f in filas[1:]]
    return Extraccion(
        "\n\n".join(b for b in bloques if b),
        {"filas": len(filas) - 1, "columnas": cabecera},
    )


def _extraer_xlsx(ruta: Path) -> Extraccion:
    import openpyxl

    try:
        wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    except Exception as e:
        return Extraccion("", error=f"xlsx ilegible: {e}")

    bloques: list[str] = []
    hojas: list[str] = []
    for ws in wb.worksheets:
        filas = list(ws.iter_rows(values_only=True))
        if not filas:
            continue
        hojas.append(ws.title)
        cabecera = [str(c) if c is not None else "" for c in filas[0]]
        bloques.append(f"Hoja: {ws.title}")
        bloques.extend(
            b for b in (_fila_a_texto(cabecera, list(f)) for f in filas[1:]) if b
        )
    wb.close()

    return Extraccion("\n\n".join(bloques), {"hojas": hojas})


# --- PBF (teselas vectoriales) -------------------------------------------


def _extraer_pbf(ruta: Path) -> Extraccion:
    """Vuelca los atributos de cada elemento del mapa como `clave: valor`.

    Un mismo elemento puede repetirse dentro de la tesela; se deduplica por el
    conjunto de atributos para no inflar el índice con copias idénticas
    para no duplicar la data.
    """
    try:
        import mapbox_vector_tile

        tile = mapbox_vector_tile.decode(ruta.read_bytes())
    except Exception as e:
        return Extraccion("", error=f"pbf ilegible: {e}")

    vistos: set[str] = set()
    bloques: list[str] = []
    capas: list[str] = []

    for nombre_capa, capa in tile.items():
        capas.append(nombre_capa)
        for elemento in capa.get("features", []):
            props = elemento.get("properties") or {}
            pares = [
                f"{k}: {v}"
                for k, v in props.items()
                if v is not None and str(v).strip() and str(v).lower() != "null"
            ]
            if not pares:
                continue
            linea = f"{nombre_capa} | " + " | ".join(pares)
            if linea in vistos:
                continue
            vistos.add(linea)
            bloques.append(linea)

    return Extraccion("\n".join(bloques), {"capas": capas, "elementos": len(bloques)})


# --- Texto plano e imágenes ----------------------------------------------


def _extraer_txt(ruta: Path) -> Extraccion:
    for encoding in ("utf-8", "latin-1"):
        try:
            return Extraccion(ruta.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return Extraccion("", error="txt ilegible")


def _extraer_imagen(ruta: Path) -> Extraccion:
    """Las imágenes se resuelven por OCR; aquí solo se marcan."""
    return Extraccion("", {"imagen": True}, necesita_ocr=True)


# --- Despacho -------------------------------------------------------------

_EXTRACTORES = {
    "pdf": _extraer_pdf,
    "json": _extraer_json,
    "csv": _extraer_csv,
    "xlsx": _extraer_xlsx,
    "pbf": _extraer_pbf,
    "txt": _extraer_txt,
    "jpg": _extraer_imagen,
    "jpeg": _extraer_imagen,
    "png": _extraer_imagen,
    "avif": _extraer_imagen,
}


def extraer(doc: Documento) -> Extraccion:
    """Extrae el texto de un documento del catálogo."""
    extractor = _EXTRACTORES.get(doc.formato)
    if extractor is None:
        return Extraccion("", error=f"formato no soportado: {doc.formato}")
    if not doc.ruta.exists():
        return Extraccion("", error="archivo no encontrado")
    try:
        return extractor(doc.ruta)
    except Exception as e:  # un documento roto no debe tumbar la ingesta
        return Extraccion("", error=f"{type(e).__name__}: {e}")
