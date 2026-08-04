"""Expansión de la consulta por realimentación de pseudo-relevancia (Rocchio).

**Por qué aquí.** El análisis del pool dice que el material relevante ya está
recuperado —recall de documento 0,998 a profundidad 200, y el 100% de las
consultas tiene algún fragmento relevante ya en el top-10— y que lo que falla es
colocarlo arriba. Rocchio ataca eso sin tocar el índice: mueve el vector de
consulta hacia el centroide de lo que ya salió primero, en la idea de que los
primeros resultados comparten el vocabulario del tema aunque la consulta no lo
nombre.

**Por qué es admisible.** La §8.3 prohíbe reformular o expandir la consulta
*mediante un decoder* y exige que la recuperación opere sobre vectores,
puntuaciones y metadata. Aquí no interviene ningún modelo: es una media de
vectores que ya están en el índice. No se genera texto en ningún punto.

**El riesgo es la deriva de tema.** Si los primeros resultados no son del tema,
el centroide arrastra la consulta hacia el error y la segunda pasada es peor que
la primera. Por eso `alfa` conserva la consulta original y `beta` pondera la
realimentación, y por eso los tests fijan que con beta=0 no se mueva nada.

    uv run pytest tests/test_realimentacion.py
"""

from __future__ import annotations

import numpy as np
import pytest

from aphelion.busqueda.recuperacion import expandir_rocchio


def unitario(*componentes: float) -> np.ndarray:
    v = np.array(componentes, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestExpandirRocchio:
    def test_sin_realimentacion_la_consulta_no_se_mueve(self):
        q = unitario(1, 0, 0)
        assert np.allclose(expandir_rocchio(q, np.empty((0, 3), np.float32)), q)

    def test_beta_cero_deja_la_consulta_intacta(self):
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(0, 1, 0), unitario(0, 0, 1)])
        assert np.allclose(expandir_rocchio(q, vecinos, beta=0.0), q)

    def test_el_resultado_es_unitario(self):
        """El índice es IndexFlatIP y solo equivale al coseno con norma 1."""
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(1, 1, 0), unitario(1, 0, 1)])
        assert np.linalg.norm(expandir_rocchio(q, vecinos, beta=0.5)) == pytest.approx(1.0)

    def test_la_consulta_se_acerca_al_centroide_de_la_realimentacion(self):
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(0, 1, 0), unitario(0, 1, 0)])
        expandida = expandir_rocchio(q, vecinos, beta=0.5)
        centroide = unitario(0, 1, 0)
        assert float(expandida @ centroide) > float(q @ centroide)

    def test_no_se_aleja_tanto_que_deje_de_parecerse_a_la_consulta(self):
        """alfa manda sobre beta: el tema original tiene que sobrevivir a la
        expansión, o la segunda pasada busca otra cosa."""
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(0, 1, 0)] * 10)
        expandida = expandir_rocchio(q, vecinos, alfa=1.0, beta=0.5)
        assert float(expandida @ q) > 0.5

    def test_beta_mayor_mueve_mas(self):
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(0, 1, 0)])
        suave = expandir_rocchio(q, vecinos, beta=0.2)
        fuerte = expandir_rocchio(q, vecinos, beta=0.8)
        assert float(fuerte @ q) < float(suave @ q)

    def test_promedia_la_realimentacion_en_vez_de_sumarla(self):
        """Sumar haría que el peso de la realimentación dependiera de cuántos
        vecinos se tomen, y `beta` dejaría de significar lo mismo con k=5 que
        con k=20."""
        q = unitario(1, 0, 0)
        uno = np.stack([unitario(0, 1, 0)])
        muchos = np.stack([unitario(0, 1, 0)] * 8)
        assert np.allclose(expandir_rocchio(q, uno, beta=0.5),
                           expandir_rocchio(q, muchos, beta=0.5))

    def test_un_centroide_que_anula_la_consulta_no_divide_por_cero(self):
        """Vecinos exactamente opuestos a la consulta con beta=1 dan el vector
        cero, que no se puede normalizar. Devolver la consulta original es lo
        correcto: sin dirección nueva, la expansión no aporta nada."""
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(-1, 0, 0)])
        assert np.allclose(expandir_rocchio(q, vecinos, alfa=1.0, beta=1.0), q)

    def test_devuelve_float32_como_espera_faiss(self):
        q = unitario(1, 0, 0)
        vecinos = np.stack([unitario(0, 1, 0)])
        assert expandir_rocchio(q, vecinos).dtype == np.float32


class TestVectoresDelIndice:
    """Rocchio necesita los vectores de los primeros resultados, y el índice
    solo devolvía posiciones y metadata."""

    def indice(self):
        from aphelion.indice import vectores as mod

        m = np.stack([unitario(1, 0, 0), unitario(0, 1, 0), unitario(0, 0, 1)])
        metadata = [{"chunk_id": f"c{i}", "doc_id": "D1", "texto": f"t{i}",
                     "fuente": "D1.pdf", "fenomeno": 1, "idioma": "es"}
                    for i in range(3)]
        return mod.IndiceVectorial("x", mod.construir(m, 3), metadata), m

    def test_reconstruye_los_vectores_por_posicion(self):
        indice, m = self.indice()
        assert np.allclose(indice.vectores([0, 2]), m[[0, 2]], atol=1e-6)

    def test_respeta_el_orden_que_se_le_pide(self):
        indice, m = self.indice()
        assert np.allclose(indice.vectores([2, 0]), m[[2, 0]], atol=1e-6)

    def test_sin_posiciones_devuelve_una_matriz_vacia_con_la_dimension(self):
        indice, _ = self.indice()
        vacia = indice.vectores([])
        assert vacia.shape == (0, 3)


class TestBuscarConRealimentacion:
    """La consulta es ambigua entre dos temas y el primer resultado desempata.

    `ancla` es lo que la consulta encuentra sola y se inclina hacia el eje x.
    `mismo_tema` y `otro_tema` son los dos temas puros: sin expansión gana
    `otro_tema` por poco, y si la realimentación funciona, el centroide del
    ancla arrastra la consulta hacia el eje x y `mismo_tema` lo adelanta.
    """

    def indice(self):
        from aphelion.indice import vectores as mod

        m = np.stack([
            unitario(1, 0.3, 0),      # ancla: lo mejor para la consulta sola
            unitario(0.98, 0, 0.199),  # mismo tema que el ancla
            unitario(0, 1, 0),         # el otro tema
        ])
        nombres = ["ancla", "mismo_tema", "otro_tema"]
        metadata = [{"chunk_id": n, "doc_id": f"D{i}", "texto": f"t{i}",
                     "fuente": f"D{i}.pdf", "fenomeno": 1, "idioma": "es"}
                    for i, n in enumerate(nombres)]
        return mod.IndiceVectorial("x", mod.construir(m, 3), metadata)

    def recuperador(self, **kw):
        from aphelion.busqueda.recuperacion import Recuperador

        rec = Recuperador({"x": self.indice()}, **kw)
        rec._encoder = lambda nombre: _EncoderFijo()
        return rec

    def orden(self, **kw):
        from aphelion.busqueda.consultas import Consulta

        c = Consulta(query_id="q001", texto="da igual")
        salida = self.recuperador(**kw).buscar(c, candidatos_por_indice=3)
        return [m["chunk_id"] for m, _ in salida["x"]]

    def test_sin_prf_gana_el_otro_tema_por_poco(self):
        assert self.orden(prf_k=0) == ["ancla", "otro_tema", "mismo_tema"]

    def test_con_prf_el_tema_del_primer_resultado_adelanta_al_otro(self):
        assert self.orden(prf_k=1, prf_beta=0.9) == ["ancla", "mismo_tema", "otro_tema"]

    def test_beta_cero_equivale_a_no_expandir(self):
        assert self.orden(prf_k=1, prf_beta=0.0) == self.orden(prf_k=0)

    def test_prf_no_cambia_cuantos_candidatos_se_devuelven(self):
        assert len(self.orden(prf_k=1, prf_beta=0.9)) == len(self.orden(prf_k=0))


class _EncoderFijo:
    """Devuelve siempre el mismo vector de consulta: el test mide la
    realimentación, no la codificación.

    La consulta reparte su peso entre los dos ejes de tema a propósito; una
    consulta que ya apuntara a un tema no dejaría margen a la expansión."""

    def codificar_consultas(self, textos):
        return np.stack([unitario(1, 1, 0) for _ in textos])
