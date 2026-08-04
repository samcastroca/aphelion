"""Comparar dos entregables reales exige el mismo rigor que comparar recetas.

`comparar.py` es lo que se ejecuta sobre los `resultados.jsonl` de verdad, que
es donde se decide qué se manda. Ordenaba por Conteo de Borda y enseñaba dos
intervalos bootstrap, y con eso no se distingue una ventaja pequeña y constante
—que gana el reto— de una grande que viene de dos consultas afortunadas.

El mismo test pareado que usa el barrido responde eso, y aquí hace todavía más
falta: entre dos corridas de la misma familia, la mayoría de las consultas dan
exactamente lo mismo y solo unas pocas se mueven. El bootstrap no pareado mide
la varianza de las cincuenta; el pareado mide solo las que cambian.

    uv run pytest tests/test_comparar.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def cargar():
    ruta = RAIZ / "scripts" / "analisis" / "comparar.py"
    spec = importlib.util.spec_from_file_location("comparar", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["comparar"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def resumen(valores: dict[str, tuple[float, float]]) -> dict:
    """Un resumen con solo lo que el test pareado necesita."""
    return {
        "por_consulta": {
            q: {"ndcg@10": n, "f1@3": f, "fenomeno": 1}
            for q, (n, f) in valores.items()
        }
    }


class TestComparacionPareada:
    def test_una_ventaja_en_todas_las_consultas_sale_significativa(self):
        mod = cargar()
        qids = [f"q{i:03d}" for i in range(20)]
        a = resumen({q: (0.7, 0.7) for q in qids})
        b = resumen({q: (0.4, 0.4) for q in qids})
        difs = mod.comparar_pareado(a, b)
        assert difs["ndcg@10"][0] == pytest.approx(0.3)
        assert difs["ndcg@10"][1] < 0.001

    def test_una_ventaja_de_una_sola_consulta_no_convence(self):
        mod = cargar()
        qids = [f"q{i:03d}" for i in range(20)]
        base = {q: (0.5, 0.5) for q in qids}
        con_suerte = dict(base)
        con_suerte["q000"] = (1.0, 1.0)
        difs = mod.comparar_pareado(resumen(con_suerte), resumen(base))
        assert difs["ndcg@10"][1] > 0.05

    def test_solo_se_parean_las_consultas_que_ambas_tienen(self):
        """Dos corridas sobre juegos de consultas distintos no son comparables
        entera a entera; parear lo común es lo único honesto que se puede hacer."""
        mod = cargar()
        a = resumen({"q001": (0.9, 0.9), "q002": (0.9, 0.9), "q003": (0.1, 0.1)})
        b = resumen({"q001": (0.5, 0.5), "q002": (0.5, 0.5)})
        difs = mod.comparar_pareado(a, b)
        assert difs["ndcg@10"][0] == pytest.approx(0.4)

    def test_sin_consultas_en_comun_no_hay_comparacion(self):
        mod = cargar()
        a = resumen({"q001": (0.9, 0.9)})
        b = resumen({"q900": (0.5, 0.5)})
        assert mod.comparar_pareado(a, b) == {}

    def test_compara_las_dos_metricas(self):
        mod = cargar()
        qids = [f"q{i:03d}" for i in range(10)]
        a = resumen({q: (0.8, 0.2) for q in qids})
        b = resumen({q: (0.2, 0.8) for q in qids})
        difs = mod.comparar_pareado(a, b)
        assert set(difs) == {"ndcg@10", "f1@3"}
        assert difs["ndcg@10"][0] > 0
        assert difs["f1@3"][0] < 0
