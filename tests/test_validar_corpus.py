"""Qué recetas se pueden medir contra unos índices ya construidos.

La validación en el corpus completo reutiliza `entrega/base_vectorial`, que
tiene los encoders de la entrega y el chunking de la entrega. Una receta que
pida otra cosa no es comparable ahí, y hay dos formas de equivocarse: medirla
igual —contra vectores de otro chunking, que da un número plausible de algo que
no es— o abortar por ella y dejar sin medir a las demás, que es lo que se vino a
hacer.

    uv run pytest tests/test_validar_corpus.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from aphelion import config
from aphelion.evaluacion.recetas import RECETAS

RAIZ = Path(__file__).resolve().parents[1]


def cargar():
    ruta = RAIZ / "scripts" / "analisis" / "validar_corpus.py"
    spec = importlib.util.spec_from_file_location("validar_corpus", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["validar_corpus"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


TODO_DISPONIBLE = lambda _e: True  # noqa: E731


class TestRecetasMedibles:
    def test_una_receta_del_chunking_de_la_entrega_se_mide(self):
        mod = cargar()
        medibles, _ = mod.recetas_medibles(
            {"entrega": RECETAS["entrega"]}, TODO_DISPONIBLE)
        assert list(medibles) == ["entrega"]

    def test_una_receta_de_otro_chunking_se_omite(self):
        """`granular` fragmenta a 256 tokens: sus vectores no son estos."""
        mod = cargar()
        medibles, omitidas = mod.recetas_medibles(
            {"granular": RECETAS["granular"]}, TODO_DISPONIBLE)
        assert medibles == {}
        assert "granular" in dict(omitidas)

    def test_una_receta_sin_indice_se_omite_sin_tumbar_a_las_demas(self):
        mod = cargar()
        sin_gte = lambda e: e != "gte-multilingual-base"  # noqa: E731
        medibles, omitidas = mod.recetas_medibles(
            {"entrega": RECETAS["entrega"], "familias": RECETAS["familias"]},
            sin_gte)
        assert list(medibles) == ["entrega"]
        assert "familias" in dict(omitidas)

    def test_cada_omision_explica_su_motivo(self):
        mod = cargar()
        _, omitidas = mod.recetas_medibles(
            {"granular": RECETAS["granular"]}, TODO_DISPONIBLE)
        assert "256" in dict(omitidas)["granular"]

    def test_el_solape_tambien_descalifica(self):
        """`granular` va sin solape; con el mismo chunk pero otro solape los
        fragmentos tampoco coinciden."""
        mod = cargar()
        from dataclasses import replace

        otra = replace(RECETAS["entrega"], solape=config.CHUNK_SOLAPE + 0.1)
        medibles, _ = mod.recetas_medibles({"x": otra}, TODO_DISPONIBLE)
        assert medibles == {}

    def test_sin_ninguna_medible_lo_dice_en_vez_de_reventar(self):
        mod = cargar()
        medibles, omitidas = mod.recetas_medibles(
            {"granular": RECETAS["granular"]}, TODO_DISPONIBLE)
        assert not medibles
        assert len(omitidas) == 1
