"""OCR para los documentos sin capa de texto.

53 PDFs del corpus carecen de capa de texto; 48 de ellos son informes escaneados
de la Defensoría del Pueblo, que son precisamente el material que responde las
consultas q033–q050. Omitirlos dejaría ciego al sistema en el tercio del examen
correspondiente al fenómeno 3.

**Encuadre normativo.** El OCR pertenece al preprocesamiento (§2.1), etapa en la
que la especificación recomienda explícitamente aplicarlo. Las prohibiciones sobre
arquitecturas decoder afectan a la generación de embeddings (§4.2) y al módulo de
recuperación (§8.3); ninguna de esas etapas emplea aquí un modelo generativo. El
uso de un VLM para OCR se declara en el informe técnico.

**Backends disponibles.** Se implementan dos para poder elegir con evidencia:

- `unlimited`: baidu/Unlimited-OCR, 3B, licencia MIT. Entiende el layout y produce
  markdown estructurado. Requiere GPU.
- `tesseract`: motor clásico. No alucina —cuando falla produce ruido evidente—
  pero ignora la estructura del documento.

El criterio de selección que domina es la **ausencia de alucinación**: un modelo
que inventa texto introduce evidencia falsa en el índice, y esa evidencia puede
acabar presentada al jurado como respuesta.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def pdf_a_imagenes(ruta: Path, dpi: int = 200, max_paginas: int | None = None):
    """Rasteriza el PDF. 200 dpi es el punto donde el OCR deja de mejorar de
    forma apreciable y la memoria todavía es razonable en documentos largos."""
    import pymupdf
    from PIL import Image

    imagenes = []
    with pymupdf.open(ruta) as doc:
        paginas = doc if max_paginas is None else list(doc)[:max_paginas]
        for pagina in paginas:
            pix = pagina.get_pixmap(dpi=dpi)
            imagenes.append(
                Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            )
    return imagenes


# --- Tesseract ------------------------------------------------------------


def _ocr_tesseract(ruta: Path, idioma: str = "spa", max_paginas=None) -> ResultadoOCR:
    import time

    try:
        import pytesseract
    except ImportError:
        return ResultadoOCR("tesseract", [], 0.0, error="pytesseract no instalado")

    t0 = time.time()
    try:
        imagenes = pdf_a_imagenes(ruta, max_paginas=max_paginas)
        paginas = [
            PaginaOCR(i, pytesseract.image_to_string(img, lang=idioma))
            for i, img in enumerate(imagenes, start=1)
        ]
    except Exception as e:
        return ResultadoOCR("tesseract", [], time.time() - t0, error=str(e))

    return ResultadoOCR("tesseract", paginas, time.time() - t0)


# --- Unlimited-OCR --------------------------------------------------------

_MODELO_UNLIMITED = "baidu/Unlimited-OCR"
_cache_unlimited: dict = {}


def _cargar_unlimited(device: str | None = None):
    if "modelo" in _cache_unlimited:
        return _cache_unlimited["modelo"], _cache_unlimited["processor"]

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(_MODELO_UNLIMITED, trust_remote_code=True)
    modelo = AutoModelForCausalLM.from_pretrained(
        _MODELO_UNLIMITED,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    modelo.eval()

    _cache_unlimited["modelo"] = modelo
    _cache_unlimited["processor"] = processor
    return modelo, processor


def _ocr_unlimited(ruta: Path, idioma: str = "spa", max_paginas=None) -> ResultadoOCR:
    import time

    t0 = time.time()
    try:
        import torch

        modelo, processor = _cargar_unlimited()
        imagenes = pdf_a_imagenes(ruta, max_paginas=max_paginas)

        paginas = []
        for i, imagen in enumerate(imagenes, start=1):
            entradas = processor(images=imagen, return_tensors="pt").to(modelo.device)
            with torch.no_grad():
                generado = modelo.generate(
                    **entradas,
                    max_new_tokens=4096,
                    # El modelo documenta este parámetro para evitar bucles de
                    # repetición, un modo de fallo propio de los VLM de OCR.
                    no_repeat_ngram_size=35,
                )
            texto = processor.batch_decode(generado, skip_special_tokens=True)[0]
            paginas.append(PaginaOCR(i, texto))
    except Exception as e:
        return ResultadoOCR(
            "unlimited", [], time.time() - t0, error=f"{type(e).__name__}: {e}"
        )

    return ResultadoOCR("unlimited", paginas, time.time() - t0)


# --- Despacho -------------------------------------------------------------

MOTORES = {"tesseract": _ocr_tesseract, "unlimited": _ocr_unlimited}


def procesar(
    ruta: Path,
    motor: str = "tesseract",
    idioma: str = "spa",
    max_paginas: int | None = None,
) -> ResultadoOCR:
    if motor not in MOTORES:
        raise KeyError(f"motor desconocido: {motor}. Disponibles: {list(MOTORES)}")
    return MOTORES[motor](Path(ruta), idioma, max_paginas)
