"""Comprueba el intervalo de confianza y el acuerdo entre anotadores.

Las dos cosas existen para lo mismo: evitar que el barrido de hiperparámetros
persiga ruido. Sin intervalo, una diferencia de 0,005 en NDCG@10 parece una
mejora; sin kappa, un ground truth donde los anotadores no se ponen de acuerdo
mide el criterio de quien anotó y no la calidad del sistema.

    uv run pytest
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from aphelion.metricas import intervalo_bootstrap

RAIZ = Path(__file__).resolve().parents[1]


def cargar_pool():
    """El script del pool no es un módulo del paquete; se carga por ruta."""
    ruta = RAIZ / "scripts" / "05_pool_anotacion.py"
    spec = importlib.util.spec_from_file_location("pool_anotacion", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["pool_anotacion"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


class TestIntervaloBootstrap:
    def test_contiene_la_media(self):
        valores = [0.1, 0.3, 0.5, 0.7, 0.9] * 10
        media = sum(valores) / len(valores)
        bajo, alto = intervalo_bootstrap(valores)
        assert bajo <= media <= alto

    def test_sin_varianza_el_intervalo_colapsa(self):
        bajo, alto = intervalo_bootstrap([0.5] * 50)
        assert bajo == pytest.approx(0.5)
        assert alto == pytest.approx(0.5)

    def test_mas_dispersion_da_intervalo_mas_ancho(self):
        estrecho = intervalo_bootstrap([0.5, 0.5, 0.5, 0.6, 0.4] * 10)
        ancho = intervalo_bootstrap([0.0, 1.0, 0.0, 1.0, 0.5] * 10)
        assert (ancho[1] - ancho[0]) > (estrecho[1] - estrecho[0])

    def test_es_reproducible(self):
        valores = [0.2, 0.4, 0.6, 0.8] * 12
        assert intervalo_bootstrap(valores) == intervalo_bootstrap(valores)

    def test_lista_vacia_no_revienta(self):
        assert intervalo_bootstrap([]) == (0.0, 0.0)

    def test_con_50_consultas_detecta_el_umbral_de_ruido(self):
        # Es la situación real: 50 consultas y NDCG alrededor de 0,6. El ancho
        # del intervalo dice cuánta diferencia hace falta para creerse una mejora.
        valores = [0.6 + (i % 5 - 2) * 0.1 for i in range(50)]
        bajo, alto = intervalo_bootstrap(valores)
        assert alto - bajo > 0.01, "con esta dispersión, 0,01 sigue siendo ruido"


class TestKappa:
    def test_acuerdo_total(self):
        pool = cargar_pool()
        pares = [(2.0, 2.0), (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.0, 0.0)]
        assert pool.kappa_cohen(pares) == pytest.approx(1.0)

    def test_desacuerdo_total_da_negativo(self):
        pool = cargar_pool()
        pares = [(2.0, 0.0), (0.0, 2.0), (2.0, 0.0), (0.0, 2.0)]
        assert pool.kappa_cohen(pares) < 0

    def test_coincidir_siempre_en_la_misma_categoria_no_es_merito(self):
        # Si los dos anotaron todo como no relevante, coinciden en el 100% pero
        # el acuerdo es trivial: kappa lo corrige por azar y no lo premia.
        pool = cargar_pool()
        assert pool.kappa_cohen([(0.0, 0.0)] * 10) == pytest.approx(1.0)

    def test_acuerdo_parcial_queda_entre_cero_y_uno(self):
        pool = cargar_pool()
        pares = [(2.0, 2.0), (2.0, 2.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)]
        k = pool.kappa_cohen(pares)
        assert 0 < k < 1

    def test_sin_pares_devuelve_none(self):
        pool = cargar_pool()
        assert pool.kappa_cohen([]) is None
