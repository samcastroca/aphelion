"""Wilcoxon de rangos con signo, contra casos con respuesta conocida.

Se implementa en el repo en vez de tomarlo de scipy —que está en el entorno solo
como dependencia transitiva de sentence-transformers— así que hay que comprobarlo
contra algo. Aquí ese algo son: los valores publicados del test para muestras
pequeñas, las propiedades que debe cumplir por construcción, y scipy cuando está
disponible, que es la referencia obvia aunque no se pueda depender de ella.

    uv run pytest tests/test_wilcoxon.py
"""

from __future__ import annotations

import pytest

from aphelion.evaluacion.metricas import p_valor_wilcoxon, _distribucion_rangos


class TestDistribucionExacta:
    def test_cuenta_todas_las_asignaciones_de_signo(self):
        """Con n rangos hay 2^n repartos, y la distribución los cuenta todos."""
        for n in (1, 5, 10):
            assert sum(_distribucion_rangos(n)) == 2**n

    def test_es_simetrica(self):
        formas = _distribucion_rangos(6)
        assert formas == formas[::-1]

    def test_caso_de_libro_n5(self):
        """Con n=5, W=0 se alcanza de una sola forma: p bilateral = 2/32."""
        _, p = p_valor_wilcoxon([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
        assert p == pytest.approx(2 / 32)


class TestPropiedades:
    def test_dos_medidas_identicas_no_deciden_nada(self):
        _, p = p_valor_wilcoxon([0.5, 0.3, 0.9], [0.5, 0.3, 0.9])
        assert p == 1.0

    def test_las_diferencias_nulas_se_descartan(self):
        """Una consulta donde ambas empatan no aporta evidencia."""
        con_empates = p_valor_wilcoxon([1, 2, 3, 0.5], [0, 0, 0, 0.5])
        sin_empates = p_valor_wilcoxon([1, 2, 3], [0, 0, 0])
        assert con_empates[1] == pytest.approx(sin_empates[1])

    def test_devuelve_la_mediana_de_las_diferencias(self):
        mediana, _ = p_valor_wilcoxon([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert mediana == pytest.approx(2.0)

    def test_una_ventaja_constante_es_significativa(self):
        a = [0.5 + 0.05 * i for i in range(20)]
        b = [0.4 + 0.05 * i for i in range(20)]
        _, p = p_valor_wilcoxon(a, b)
        assert p < 0.001

    def test_una_ventaja_de_una_sola_consulta_no_lo_es(self):
        """Donde más se separa de la permutación: Wilcoxon ignora la magnitud."""
        a = [0.5] * 19 + [0.99]
        b = [0.5] * 19 + [0.01]
        _, p = p_valor_wilcoxon(a, b)
        assert p > 0.5

    def test_exige_las_mismas_consultas(self):
        with pytest.raises(ValueError, match="pareado"):
            p_valor_wilcoxon([1, 2, 3], [1, 2])

    def test_sin_consultas_es_un_error_y_no_un_cero(self):
        with pytest.raises(ValueError):
            p_valor_wilcoxon([], [])


class TestContraScipy:
    """scipy no es dependencia declarada; si está, se usa como referencia."""

    @pytest.fixture(autouse=True)
    def _scipy(self):
        self.stats = pytest.importorskip("scipy.stats")

    @pytest.mark.parametrize("caso", [
        ([0.71, 0.55, 0.62, 0.90, 0.31], [0.68, 0.55, 0.70, 0.80, 0.42]),
        ([0.1 * i for i in range(12)], [0.1 * i + 0.03 for i in range(12)]),
        ([0.5, 0.5, 0.6, 0.7, 0.2, 0.9, 0.4], [0.4, 0.6, 0.6, 0.5, 0.3, 0.8, 0.4]),
    ])
    def test_coincide_con_la_referencia(self, caso):
        a, b = caso
        _, mio = p_valor_wilcoxon(a, b)
        difs = [x - y for x, y in zip(a, b)]
        hay_empates = len({abs(d) for d in difs if d}) < len([d for d in difs if d])
        suyo = self.stats.wilcoxon(
            a, b,
            # scipy usa la exacta sin empates y la normal con ellos, igual que
            # esta implementación; se le pide explícitamente el mismo modo.
            method="approx" if hay_empates else "exact",
            correction=hay_empates,
            zero_method="wilcox",
        ).pvalue
        assert mio == pytest.approx(suyo, abs=1e-9)
