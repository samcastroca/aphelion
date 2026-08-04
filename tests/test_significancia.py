"""El test que decide si una receta le gana a otra tiene que ser pareado.

Todas las recetas se miden sobre las **mismas** consultas, así que comparar dos
intervalos bootstrap independientes tira la mayor parte de la información: una
consulta difícil hunde a las dos por igual y aparece como varianza en ambas,
cuando en realidad no dice nada sobre cuál es mejor. El test pareado la anula.

Además, el bootstrap tiene sesgo documentado hacia p-valores pequeños en
conjuntos de consultas pequeños —el nuestro son 50—, lo que infla los falsos
positivos. La permutación calcula el p-valor exacto por construcción: bajo la
hipótesis nula las dos recetas son intercambiables consulta a consulta, así que
cambiar el signo de una diferencia es una reasignación válida.

    uv run pytest tests/test_significancia.py
"""

from __future__ import annotations

import pytest

from aphelion.evaluacion import metricas


class TestPValorPermutacion:
    def test_dos_medidas_identicas_no_se_distinguen(self):
        a = [0.5, 0.2, 0.9, 0.1]
        dif, p = metricas.p_valor_permutacion(a, list(a))
        assert dif == 0.0
        assert p == pytest.approx(1.0)

    def test_una_ventaja_sistematica_sale_significativa(self):
        # Gana en las 20 consultas: no hay reasignación de signos que produzca
        # una diferencia media tan grande salvo la original.
        a = [0.6] * 20
        b = [0.4] * 20
        dif, p = metricas.p_valor_permutacion(a, b, n=20_000)
        assert dif == pytest.approx(0.2)
        assert p < 0.001

    def test_una_ventaja_en_ruido_no_sale_significativa(self):
        a = [0.5, 0.6, 0.4, 0.7, 0.3, 0.55, 0.45, 0.65]
        b = [0.6, 0.5, 0.5, 0.6, 0.4, 0.50, 0.50, 0.60]
        _, p = metricas.p_valor_permutacion(a, b, n=20_000)
        assert p > 0.05

    def test_el_signo_de_la_diferencia_sigue_el_orden_de_los_argumentos(self):
        a, b = [0.2, 0.3], [0.5, 0.4]
        dif_ab, p_ab = metricas.p_valor_permutacion(a, b)
        dif_ba, p_ba = metricas.p_valor_permutacion(b, a)
        assert dif_ab == pytest.approx(-dif_ba)
        assert p_ab == pytest.approx(p_ba)

    def test_las_consultas_empatadas_no_mueven_el_p_valor(self):
        """Un empate aporta una diferencia de cero, y cero cambia de signo sin
        cambiar nada. Añadir empates no puede volver significativo lo que no lo
        era, ni al revés; solo diluye la media."""
        a = [0.9, 0.8, 0.7]
        b = [0.5, 0.4, 0.3]
        _, p_solo = metricas.p_valor_permutacion(a, b, n=20_000)
        _, p_con_empates = metricas.p_valor_permutacion(
            a + [0.5] * 10, b + [0.5] * 10, n=20_000
        )
        assert p_solo == pytest.approx(p_con_empates, abs=0.01)

    def test_la_misma_semilla_da_el_mismo_p_valor(self):
        a = [0.5, 0.2, 0.9, 0.1, 0.4]
        b = [0.4, 0.3, 0.7, 0.2, 0.5]
        primero = metricas.p_valor_permutacion(a, b, n=5_000, semilla=7)
        segundo = metricas.p_valor_permutacion(a, b, n=5_000, semilla=7)
        assert primero == segundo

    def test_el_p_valor_nunca_es_cero(self):
        """Un p-valor de 0 afirmaría certeza que un test de remuestreo no tiene.
        La corrección de continuidad (+1 arriba y abajo) lo acota por abajo."""
        a = [1.0] * 30
        b = [0.0] * 30
        _, p = metricas.p_valor_permutacion(a, b, n=1_000)
        assert 0 < p <= 1

    def test_listas_de_distinta_longitud_son_un_error(self):
        with pytest.raises(ValueError):
            metricas.p_valor_permutacion([0.1, 0.2], [0.1])

    def test_sin_consultas_no_hay_nada_que_comparar(self):
        with pytest.raises(ValueError):
            metricas.p_valor_permutacion([], [])
