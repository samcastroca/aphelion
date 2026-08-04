"""Reparto de la culpa entre recuperar y ordenar.

Una métrica baja tiene dos causas que se arreglan con cosas distintas: o lo
relevante no entra en el pool —y entonces hay que tocar el encoder, el chunking
o la profundidad— o entra y queda mal colocado, y entonces lo que hay que tocar
es la fusión, el realce y la agregación. Confundirlas cuesta días de GPU
mejorando lo que no falla.

    uv run pytest tests/test_techo.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def cargar_techo():
    ruta = RAIZ / "scripts" / "analisis" / "techo.py"
    spec = importlib.util.spec_from_file_location("techo", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["techo"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


class TestRecall:
    def test_traerlos_todos_es_recall_uno(self):
        mod = cargar_techo()
        assert mod.recall(["a", "b", "c"], {"a", "b"}) == pytest.approx(1.0)

    def test_traer_la_mitad_es_recall_medio(self):
        mod = cargar_techo()
        assert mod.recall(["a", "z"], {"a", "b"}) == pytest.approx(0.5)

    def test_no_traer_ninguno_es_cero(self):
        mod = cargar_techo()
        assert mod.recall(["x", "y"], {"a", "b"}) == 0.0

    def test_repetir_un_relevante_no_lo_cuenta_dos_veces(self):
        """El pool trae fragmentos, y varios pueden ser del mismo documento."""
        mod = cargar_techo()
        assert mod.recall(["a", "a", "a"], {"a", "b"}) == pytest.approx(0.5)

    def test_sin_relevantes_no_hay_recall_definido(self):
        mod = cargar_techo()
        assert mod.recall(["a"], set()) is None


class TestVeredicto:
    """La conclusión que el diagnóstico tiene que dejar escrita."""

    def test_recall_alto_y_metrica_baja_senalan_al_ordenamiento(self):
        mod = cargar_techo()
        assert mod.veredicto(recall=0.99, metrica=0.45) == "ordenamiento"

    def test_recall_bajo_senala_a_la_recuperacion(self):
        mod = cargar_techo()
        assert mod.veredicto(recall=0.55, metrica=0.45) == "recuperación"

    def test_si_la_metrica_ya_alcanza_al_recall_no_queda_margen(self):
        mod = cargar_techo()
        assert mod.veredicto(recall=0.60, metrica=0.58) == "sin margen"
