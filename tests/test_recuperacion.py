"""Comprueba la agregación por fuente y el post-filtro de umbral relativo.

Las dos piezas existen por el mismo motivo: la evaluación no mide lo que el
sistema cree devolver, sino lo que el jurado empareja. Los documentos se
emparejan por `fuente` (§10.2.1) y el corpus tiene 59 nombres repetidos en 186
archivos, así que dos doc_id con la misma fuente en el top-3 son un solo acierto
posible. El umbral es el post-filtro de la §8.7, relativo porque las escalas de
coseno de los dos encoders no son comparables.

    uv run pytest
"""

from __future__ import annotations

import pytest

from aphelion.busqueda.recuperacion import (
    Candidato,
    agregar_a_documentos,
    filtrar_por_umbral_relativo,
    fusionar_rrf,
)


def candidato(chunk_id: str, doc_id: str, fuente: str, puntaje: float) -> Candidato:
    return Candidato(
        chunk_id=chunk_id,
        doc_id=doc_id,
        texto=f"texto de {chunk_id}",
        fuente=fuente,
        fenomeno=1,
        puntaje=puntaje,
        posiciones={"bge-m3": 1},
    )


class TestAgregacionPorFuente:
    def test_dos_doc_id_con_la_misma_fuente_ocupan_una_posicion(self):
        # 59 nombres estandarizados están repetidos en el corpus. Para el F1@3
        # son el mismo documento: dejar que ocupen dos de las tres posiciones
        # regala una al azar.
        cands = [
            candidato("c1", "F1-A-001", "informe.pdf", 0.9),
            candidato("c2", "F3-B-007", "informe.pdf", 0.8),  # misma fuente
            candidato("c3", "F1-C-002", "otro.pdf", 0.7),
            candidato("c4", "F1-D-003", "tercero.pdf", 0.6),
        ]
        docs = agregar_a_documentos(cands, top=3)
        assert docs == ["F1-A-001", "F1-C-002", "F1-D-003"]

    def test_la_fuente_repetida_reporta_el_doc_id_de_su_mejor_fragmento(self):
        cands = [
            candidato("c1", "F3-B-007", "informe.pdf", 0.5),
            candidato("c2", "F1-A-001", "informe.pdf", 0.9),
        ]
        docs = agregar_a_documentos(cands, top=3)
        assert docs[0] == "F1-A-001"

    def test_max_pooling_sigue_mandando(self):
        # Un documento con muchos fragmentos flojos no desplaza a uno con un
        # solo fragmento fuerte.
        cands = [candidato(f"d{i}", "DEBIL", "debil.pdf", 0.3) for i in range(5)]
        cands.append(candidato("f1", "FUERTE", "fuerte.pdf", 0.8))
        docs = agregar_a_documentos(cands, top=2)
        assert docs[0] == "FUERTE"

    def test_sin_fuente_cae_al_doc_id(self):
        cands = [
            candidato("c1", "D1", "", 0.9),
            candidato("c2", "D2", "", 0.8),
        ]
        assert agregar_a_documentos(cands, top=2) == ["D1", "D2"]


class TestUmbralRelativo:
    def _ranking(self, similitudes: list[float]) -> list[tuple[dict, float]]:
        return [
            ({"chunk_id": f"c{i}", "doc_id": f"D{i}", "texto": "t", "fuente": "f"}, s)
            for i, s in enumerate(similitudes)
        ]

    def test_descarta_la_cola_bajo_el_umbral(self):
        ranking = self._ranking([0.8, 0.76, 0.5, 0.3])
        filtrado = filtrar_por_umbral_relativo(ranking, 0.9)  # corte en 0.72
        assert [s for _, s in filtrado] == [0.8, 0.76]

    def test_es_relativo_al_mejor_no_absoluto(self):
        # La misma consulta con similitudes globalmente bajas conserva su top:
        # el corte escala con el mejor candidato.
        ranking = self._ranking([0.4, 0.38, 0.2])
        filtrado = filtrar_por_umbral_relativo(ranking, 0.9)  # corte en 0.36
        assert [s for _, s in filtrado] == [0.4, 0.38]

    def test_none_lo_desactiva(self):
        ranking = self._ranking([0.8, 0.1])
        assert filtrar_por_umbral_relativo(ranking, None) == ranking

    def test_ranking_vacio_no_revienta(self):
        assert filtrar_por_umbral_relativo([], 0.9) == []

    def test_sin_senal_positiva_no_filtra(self):
        # Con el mejor coseno en cero o negativo no hay escala utilizable.
        ranking = self._ranking([0.0, -0.2])
        assert filtrar_por_umbral_relativo(ranking, 0.9) == ranking

    def test_no_reordena_lo_que_conserva(self):
        ranking = self._ranking([0.9, 0.85, 0.84])
        filtrado = filtrar_por_umbral_relativo(ranking, 0.9)
        assert filtrado == ranking[: len(filtrado)]


class TestIdiomaDelCandidato:
    def test_la_fusion_propaga_el_idioma_de_la_metadata(self):
        # El recorte a 250 palabras segmenta con el idioma del fragmento; si la
        # fusión lo perdiera, todo se segmentaría como español.
        meta = {
            "chunk_id": "c1",
            "doc_id": "D1",
            "texto": "t",
            "fuente": "f.pdf",
            "fenomeno": 1,
            "idioma": "en",
        }
        candidatos = fusionar_rrf({"bge-m3": [(meta, 0.9)]})
        assert candidatos[0].idioma == "en"

    def test_metadata_sin_idioma_cae_a_espanol(self):
        meta = {"chunk_id": "c1", "doc_id": "D1", "texto": "t", "fuente": "f.pdf"}
        candidatos = fusionar_rrf({"bge-m3": [(meta, 0.9)]})
        assert candidatos[0].idioma == "es"
