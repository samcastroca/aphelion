"""Comprueba de dónde salen los documentos pendientes de OCR.

Salían de una lista que `01_extraer.py` escribía con lo que había procesado en
esa corrida. Con la extracción cacheada —el caso normal al reanudar— no procesa
nada, la lista quedaba vacía y la etapa de OCR moría sin entrada. Los pendientes
son un hecho sobre la caché de extracción, y de ahí se leen.

    uv run pytest
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def cargar_ocr():
    ruta = RAIZ / "scripts" / "etapas" / "02_ocr.py"
    spec = importlib.util.spec_from_file_location("etapa_ocr", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["etapa_ocr"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


etapa_ocr = cargar_ocr()


@pytest.fixture
def texto_crudo(tmp_path, monkeypatch):
    """Sustituye `trabajo/texto/` por una carpeta de mentira."""
    carpeta = tmp_path / "texto"
    carpeta.mkdir()
    monkeypatch.setattr(etapa_ocr.config, "TEXTO_CRUDO", carpeta)
    return carpeta


def escribir(carpeta: Path, doc_id: str, necesita_ocr: bool, texto: str = "") -> None:
    (carpeta / f"{doc_id}.json").write_text(
        json.dumps({"doc_id": doc_id, "texto": texto, "necesita_ocr": necesita_ocr}),
        encoding="utf-8",
    )


class TestPendientes:
    def test_los_encuentra_aunque_la_extraccion_estuviera_cacheada(self, texto_crudo):
        # Ninguna corrida los "acaba de detectar": están en disco de antes. Es
        # exactamente el caso que rompía el pipeline al reanudarlo.
        escribir(texto_crudo, "DOC-001", necesita_ocr=False, texto="con capa de texto")
        escribir(texto_crudo, "DOC-002", necesita_ocr=True)
        escribir(texto_crudo, "DOC-003", necesita_ocr=True)

        assert etapa_ocr.pendientes() == ["DOC-002", "DOC-003"]

    def test_sin_pendientes_devuelve_lista_vacia(self, texto_crudo):
        escribir(texto_crudo, "DOC-001", necesita_ocr=False, texto="algo")

        assert etapa_ocr.pendientes() == []

    def test_un_documento_ya_reconocido_deja_de_estar_pendiente(self, texto_crudo):
        # 02_ocr pone `necesita_ocr` en False al aplicar el texto reconocido, así
        # que reanudar no vuelve a pasarle Tesseract por encima.
        escribir(texto_crudo, "DOC-002", necesita_ocr=True)
        assert etapa_ocr.pendientes() == ["DOC-002"]

        escribir(texto_crudo, "DOC-002", necesita_ocr=False, texto="ya reconocido")
        assert etapa_ocr.pendientes() == []

    def test_el_orden_no_depende_del_sistema_de_archivos(self, texto_crudo):
        for doc_id in ("DOC-030", "DOC-004", "DOC-017"):
            escribir(texto_crudo, doc_id, necesita_ocr=True)

        assert etapa_ocr.pendientes() == ["DOC-004", "DOC-017", "DOC-030"]
