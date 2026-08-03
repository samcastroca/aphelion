"""OCR para los documentos sin capa de texto.

60 PDFs del corpus carecen de capa de texto; 48 de ellos son informes escaneados
de la Defensoría del Pueblo, que son precisamente el material que responde las
consultas q033–q050. Omitirlos dejaría ciego al sistema en el tercio del examen
correspondiente al fenómeno 3.

**Por qué Tesseract y no un VLM.** Se evaluó `baidu/Unlimited-OCR` (3B, MIT), que
entiende el layout y produce markdown estructurado. Se descartó por dos razones,
en este orden:

1. Es una arquitectura decoder, y el reto las prohíbe en la construcción del
   índice. El OCR es preprocesamiento y el argumento de que queda fuera del
   alcance es defendible, pero el texto que produce termina indexado y no vale
   la pena arriesgar la entrega por eso.
2. Un VLM que alucina introduce evidencia falsa en el índice, y esa evidencia
   puede acabar presentada al jurado como respuesta. Tesseract, cuando falla,
   produce ruido evidente.

La ganancia de estructura no compensa ninguno de los dos riesgos.

Requiere el binario de Tesseract instalado en el sistema, además del paquete
`pytesseract`. En Windows: `winget install UB-Mannheim.TesseractOCR` y añadir el
paquete de idioma español (`spa`).
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

MOTOR = "tesseract"

# Los 60 documentos pendientes no son todos españoles: 48 son informes de la
# Defensoría, pero el resto son de CSET y CSIS, en inglés. Tesseract acepta
# varios idiomas a la vez y decide por página, lo que evita tener que clasificar
# los documentos antes de leerlos.
IDIOMA = "spa+eng"
IDIOMAS_REQUERIDOS = ("spa", "eng")
DPI = 200

# Ubicaciones habituales del binario en Windows: el instalador de UB-Mannheim no
# añade Tesseract al PATH, así que buscarlo evita un fallo por configuración.
_RUTAS_BINARIO = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)

# Sin permisos de administrador no se puede escribir en el tessdata del
# instalador, así que el paquete de español puede vivir en el perfil de usuario.
_TESSDATA_USUARIO = Path(os.environ.get("LOCALAPPDATA", "")) / "tessdata"


def _configurar() -> None:
    """Localiza binario y datos de idioma. Idempotente y sin efectos si ya están."""
    try:
        import pytesseract
    except ImportError:
        return

    if not shutil.which("tesseract"):
        for ruta in _RUTAS_BINARIO:
            if ruta.exists():
                pytesseract.pytesseract.tesseract_cmd = str(ruta)
                break

    if "TESSDATA_PREFIX" not in os.environ:
        faltan = [
            i
            for i in IDIOMAS_REQUERIDOS
            if not (_TESSDATA_USUARIO / f"{i}.traineddata").exists()
        ]
        if not faltan:
            os.environ["TESSDATA_PREFIX"] = str(_TESSDATA_USUARIO)


@dataclass
class PaginaOCR:
    numero: int
    texto: str


@dataclass
class ResultadoOCR:
    motor: str
    paginas: list[PaginaOCR]
    segundos: float
    error: str | None = None

    @property
    def texto(self) -> str:
        return "\n\n".join(p.texto for p in self.paginas if p.texto.strip())

    @property
    def num_paginas(self) -> int:
        return len(self.paginas)


# Formatos que se entregan a Tesseract como imagen directa, sin rasterizar.
# PyMuPDF abre JPG y PNG como documentos, pero no AVIF; Pillow (>= 11.2) abre
# los cuatro, así que las imágenes van todas por Pillow y PyMuPDF queda solo
# para los PDF, que es lo suyo.
_EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".avif", ".webp", ".tif", ".tiff"}


def pdf_a_imagenes(ruta: Path, dpi: int = DPI, max_paginas: int | None = None):
    """Rasteriza el PDF. 200 dpi es el punto donde el OCR deja de mejorar de
    forma apreciable y la memoria todavía es razonable en documentos largos.

    Las imágenes sueltas no se rasterizan: se abren con Pillow y se devuelven
    como una única página."""
    import pymupdf
    from PIL import Image

    if ruta.suffix.lower() in _EXTENSIONES_IMAGEN:
        return [Image.open(ruta).convert("RGB")]

    imagenes = []
    with pymupdf.open(ruta) as doc:
        paginas = doc if max_paginas is None else list(doc)[:max_paginas]
        for pagina in paginas:
            pix = pagina.get_pixmap(dpi=dpi)
            imagenes.append(
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            )
    return imagenes


def disponible() -> tuple[bool, str]:
    """(listo, versión_o_motivo). Se comprueba antes de lanzar el lote: fallar en
    el documento 1 de 60 es mejor que en el 59.

    Verifica también que el paquete de idioma esté presente. Tesseract con solo
    `eng` no falla al pedirle `spa`: devuelve texto degradado, que es el modo de
    fallo silencioso que este proyecto no puede permitirse.
    """
    _configurar()
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract no instalado (uv add pytesseract)"

    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as e:
        return False, f"binario de tesseract no encontrado: {e}"

    try:
        idiomas = pytesseract.get_languages()
    except Exception:
        idiomas = []

    faltan = [i for i in IDIOMAS_REQUERIDOS if idiomas and i not in idiomas]
    if faltan:
        return False, (
            f"tesseract {version} sin los paquetes {faltan} (tiene {idiomas}). "
            f"Descárgalos a {_TESSDATA_USUARIO} o al tessdata de la instalación"
        )

    return True, f"{version}, idiomas {idiomas or '?'}"


def procesar(
    ruta: Path,
    idioma: str = IDIOMA,
    max_paginas: int | None = None,
    dpi: int = DPI,
) -> ResultadoOCR:
    """Extrae el texto de un PDF escaneado, página a página."""
    _configurar()
    t0 = time.time()
    try:
        import pytesseract
    except ImportError:
        return ResultadoOCR(MOTOR, [], 0.0, error="pytesseract no instalado")

    try:
        imagenes = pdf_a_imagenes(Path(ruta), dpi=dpi, max_paginas=max_paginas)
        paginas = [
            PaginaOCR(i, pytesseract.image_to_string(img, lang=idioma))
            for i, img in enumerate(imagenes, start=1)
        ]
    except Exception as e:
        return ResultadoOCR(MOTOR, [], time.time() - t0, error=f"{type(e).__name__}: {e}")

    return ResultadoOCR(MOTOR, paginas, time.time() - t0)
