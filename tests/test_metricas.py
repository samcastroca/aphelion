"""Comprueba NDCG@10 y F1@3 contra valores calculados a mano.

Estas dos funciones deciden cada barrido de hiperparámetros que vendrá después.
Si estuvieran mal, el equipo optimizaría en la dirección equivocada sin que nada
lo delatara: los números seguirían subiendo y bajando de forma plausible. De ahí
que los casos de abajo usen valores derivados a mano de las fórmulas de la §10.2
en lugar de la salida del propio código.

    uv run pytest
"""

from __future__ import annotations

import math

import pytest

from aphelion.evaluacion.metricas import Juicio, dcg, evaluar, f1_at_k, ndcg_at_k


class TestDCG:
    def test_posicion_uno_no_se_descuenta(self):
        # log2(1+1) = 1, de modo que el primer puesto vale su relevancia entera.
        assert dcg([3.0]) == pytest.approx(3.0)

    def test_descuento_logaritmico(self):
        # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5
        esperado = 3.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        assert dcg([3.0, 2.0, 1.0]) == pytest.approx(esperado)

    def test_el_orden_importa(self):
        assert dcg([1.0, 3.0]) < dcg([3.0, 1.0])


class TestNDCG:
    def test_ranking_perfecto_da_uno(self):
        relevancias = [2.0, 2.0, 1.0]
        assert ndcg_at_k(relevancias, relevancias, k=10) == pytest.approx(1.0)

    def test_ranking_invertido_penaliza(self):
        ideal = [2.0, 1.0]
        assert ndcg_at_k([1.0, 2.0], ideal, k=10) < 1.0

    def test_sin_relevantes_da_cero(self):
        # Sin nada relevante el IDCG es 0 y la división no está definida; la
        # convención es 0.0, no una excepción.
        assert ndcg_at_k([0.0, 0.0], [], k=10) == 0.0

    def test_el_corte_en_k_se_respeta(self):
        # El fragmento relevante cae en la posición 11 y no debe puntuar.
        obtenidas = [0.0] * 10 + [3.0]
        assert ndcg_at_k(obtenidas, [3.0], k=10) == 0.0

    def test_ideal_se_ordena_aunque_llegue_desordenado(self):
        # El IDCG usa el mejor orden posible, no el orden en que se anotó.
        assert ndcg_at_k([2.0, 1.0], [1.0, 2.0], k=10) == pytest.approx(1.0)


class TestF1:
    def test_tres_de_tres(self):
        assert f1_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_ninguno(self):
        assert f1_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_normalizacion_del_recall(self):
        # Un solo documento relevante y el sistema lo devuelve:
        #   P = 1/3, R = 1/min(1,3) = 1  ->  F1 = 2*(1/3)*1 / (1/3 + 1) = 0.5
        # Sin la normalización min(|D*|,3) el recall seguiría siendo 1 aquí,
        # pero el caso de abajo es el que la distingue.
        assert f1_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(0.5)

    def test_mas_relevantes_que_posiciones(self):
        # Cinco relevantes y el sistema acierta los tres que puede devolver.
        #   P = 3/3 = 1
        #   R = 3/min(5,3) = 1     <- con |D*| sin acotar sería 3/5 = 0.6
        #   F1 = 1.0               <- y el techo quedaría en 0.75
        relevantes = {"a", "b", "c", "d", "e"}
        assert f1_at_k(["a", "b", "c"], relevantes, k=3) == pytest.approx(1.0)

    def test_es_de_conjunto_no_de_orden(self):
        relevantes = {"a", "b"}
        assert f1_at_k(["a", "b", "z"], relevantes, k=3) == f1_at_k(
            ["z", "b", "a"], relevantes, k=3
        )

    def test_documentos_repetidos_no_suman(self):
        # El relleno del esquema puede repetir un doc_id cuando faltan
        # candidatos. Repetir un acierto no debe contarlo dos veces: las tres
        # posiciones con el mismo documento valen lo mismo que una acertada y
        # dos falladas.
        relevantes = {"a", "b"}
        repetido = f1_at_k(["a", "a", "a"], relevantes, k=3)
        assert repetido == f1_at_k(["a", "x", "y"], relevantes, k=3)
        # P = 1/3, R = 1/2  ->  F1 = 2*(1/3)*(1/2) / (1/3 + 1/2) = 0.4
        assert repetido == pytest.approx(0.4)


class TestEvaluar:
    def _resultado(self, query_id, chunks, docs):
        return {
            "query_id": query_id,
            "documents": [{"rank": i, "doc_id": d} for i, d in enumerate(docs, 1)],
            "fragments": [
                {"rank": i, "chunk_id": c, "doc_id": "D", "text": ""}
                for i, c in enumerate(chunks, 1)
            ],
        }

    def test_promedia_sobre_las_consultas_anotadas(self):
        juicios = {
            "q001": Juicio("q001", {"c1": 2.0}, {"d1"}),
            "q002": Juicio("q002", {"c9": 2.0}, {"d9"}),
        }
        resultados = [
            self._resultado("q001", ["c1"], ["d1", "x", "y"]),  # perfecto
            self._resultado("q002", ["zz"], ["x", "y", "z"]),  # todo mal
        ]
        salida = evaluar(resultados, juicios)

        assert salida["consultas_evaluadas"] == 2
        assert salida["ndcg@10"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert salida["f1@3"] == pytest.approx(0.25)  # (0.5 + 0.0) / 2

    def test_las_consultas_sin_anotar_no_puntuan(self):
        # No anotada no es lo mismo que fallada: promediar un 0 por cada consulta
        # sin anotar haría parecer peor a un sistema mientras crece el pool.
        juicios = {"q001": Juicio("q001", {"c1": 2.0}, {"d1"})}
        resultados = [
            self._resultado("q001", ["c1"], ["d1", "x", "y"]),
            self._resultado("q050", ["c1"], ["d1", "x", "y"]),
        ]
        salida = evaluar(resultados, juicios)

        assert salida["consultas_evaluadas"] == 1
        assert salida["ndcg@10"] == pytest.approx(1.0)
