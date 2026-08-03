"""Comprueba la maquinaria del barrido y la fusión alternativa.

El barrido decide con qué configuración se entrega, así que sus dos piezas de
juicio —el Conteo de Borda y la detección de empates— tienen que estar bien. Un
Borda mal calculado elegiría la configuración equivocada sin que nada avisara.

    uv run pytest
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from aphelion.busqueda.recuperacion import Candidato, fusionar_combsum, fusionar_rrf

RAIZ = Path(__file__).resolve().parents[1]


def cargar_barrido():
    ruta = RAIZ / "scripts" / "analisis" / "barrido.py"
    spec = importlib.util.spec_from_file_location("barrido", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["barrido"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def meta(chunk_id: str, doc_id: str = "D1", texto: str = "hola") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "texto": texto,
        "fuente": f"{doc_id}.pdf",
        "fenomeno": 1,
    }


def corrida(mod, ndcg, f1, ic_ndcg=None, ic_f1=None):
    return mod.Corrida(
        cfg={
            "fusion": "rrf",
            "k0": 60,
            "boost": 1.0,
            "max_por_doc": 3,
            "candidatos": 200,
            "subdividir": False,
        },
        ndcg=ndcg,
        f1=f1,
        ic_ndcg=ic_ndcg or (ndcg - 0.01, ndcg + 0.01),
        ic_f1=ic_f1 or (f1 - 0.01, f1 + 0.01),
    )


class TestCombSum:
    def test_suma_las_similitudes_de_ambos_indices(self):
        rankings = {
            "a": [(meta("c1"), 0.9), (meta("c2"), 0.5)],
            "b": [(meta("c2"), 0.8), (meta("c1"), 0.4)],
        }
        fusionados = {c.chunk_id: c.puntaje for c in fusionar_combsum(rankings)}
        assert fusionados["c1"] == pytest.approx(1.3)
        assert fusionados["c2"] == pytest.approx(1.3)

    def test_ausente_en_un_indice_suma_cero_por_ese_indice(self):
        rankings = {"a": [(meta("c1"), 0.9)], "b": [(meta("c2"), 0.8)]}
        fusionados = {c.chunk_id: c.puntaje for c in fusionar_combsum(rankings)}
        assert fusionados["c1"] == pytest.approx(0.9)
        assert fusionados["c2"] == pytest.approx(0.8)

    def test_conserva_la_magnitud_que_rrf_descarta(self):
        # El mismo orden en los dos índices, pero con similitudes muy distintas.
        # RRF los empata porque solo mira posiciones; CombSUM los separa.
        rankings = {
            "a": [(meta("c1"), 0.95), (meta("c2"), 0.20)],
            "b": [(meta("c1"), 0.93), (meta("c2"), 0.18)],
        }
        por_rrf = fusionar_rrf(rankings)
        por_suma = fusionar_combsum(rankings)

        brecha_rrf = por_rrf[0].puntaje - por_rrf[1].puntaje
        brecha_suma = por_suma[0].puntaje - por_suma[1].puntaje
        assert brecha_suma > brecha_rrf

    def test_ambas_devuelven_el_mismo_conjunto(self):
        rankings = {
            "a": [(meta("c1"), 0.9), (meta("c2"), 0.5)],
            "b": [(meta("c3"), 0.8)],
        }
        assert {c.chunk_id for c in fusionar_rrf(rankings)} == {
            c.chunk_id for c in fusionar_combsum(rankings)
        }


class TestBorda:
    def test_ganar_ambas_tablas_da_el_maximo(self):
        mod = cargar_barrido()
        corridas = [corrida(mod, 0.9, 0.9), corrida(mod, 0.5, 0.5), corrida(mod, 0.1, 0.1)]
        puntos = mod.puntos_borda(corridas)
        assert puntos[0] > puntos[1] > puntos[2]
        assert puntos[0] == 4  # (3-1) en cada una de las dos tablas

    def test_premia_la_consistencia_sobre_ganar_una_sola(self):
        # La primera es segunda en ambas; la segunda gana NDCG y pierde F1.
        # Borda favorece a la consistente, igual que en la clasificación del reto.
        mod = cargar_barrido()
        consistente = corrida(mod, 0.72, 0.68)
        especialista = corrida(mod, 0.81, 0.55)
        tercera = corrida(mod, 0.65, 0.62)
        puntos = mod.puntos_borda([consistente, especialista, tercera])
        assert puntos[0] > puntos[1]

    def test_una_sola_configuracion_no_revienta(self):
        mod = cargar_barrido()
        assert mod.puntos_borda([corrida(mod, 0.5, 0.5)]) == {0: 0}


class TestEmpates:
    def test_intervalos_solapados_son_indistinguibles(self):
        mod = cargar_barrido()
        a = corrida(mod, 0.700, 0.600, (0.65, 0.75), (0.55, 0.65))
        b = corrida(mod, 0.705, 0.605, (0.66, 0.76), (0.56, 0.66))
        assert mod.indistinguibles(a, b)

    def test_intervalos_separados_si_se_distinguen(self):
        mod = cargar_barrido()
        a = corrida(mod, 0.90, 0.90, (0.88, 0.92), (0.88, 0.92))
        b = corrida(mod, 0.40, 0.40, (0.38, 0.42), (0.38, 0.42))
        assert not mod.indistinguibles(a, b)

    def test_hace_falta_solape_en_las_dos_metricas(self):
        # Solapan en NDCG pero no en F1: son distinguibles.
        mod = cargar_barrido()
        a = corrida(mod, 0.70, 0.90, (0.65, 0.75), (0.88, 0.92))
        b = corrida(mod, 0.71, 0.40, (0.66, 0.76), (0.38, 0.42))
        assert not mod.indistinguibles(a, b)


class TestRejilla:
    def test_genera_el_producto_completo(self):
        mod = cargar_barrido()
        rejilla = {"a": [1, 2], "b": ["x", "y", "z"]}
        combos = mod.combinaciones(rejilla)
        assert len(combos) == 6
        assert {"a": 1, "b": "x"} in combos
        assert all(set(c) == {"a", "b"} for c in combos)

    def test_la_rejilla_completa_es_manejable(self):
        mod = cargar_barrido()
        assert len(mod.combinaciones(mod.REJILLA)) <= 250
        assert len(mod.combinaciones(mod.REJILLA_RAPIDA)) <= 20
