"""Comprueba la construcción del archivo de resultados contra la §9.

El evaluador automático descarta objetos con arrays de tamaño distinto de 3 y 10,
o con fragmentos por encima de 250 palabras (§9.3.2). Estas comprobaciones son
baratas y el fallo que previenen es caro: una entrega descartada por formato.

    uv run pytest
"""

from __future__ import annotations

import pytest

from aphelion import config
from aphelion.busqueda.recuperacion import Candidato
from aphelion.busqueda.salida import construir_fragmentos, resultado_a_dict, subdividir
from aphelion.busqueda.recuperacion import Resultado

LIMITE = config.MAX_PALABRAS_FRAGMENTO


def oraciones(n: int, palabras: int = 20, prefijo: str = "p") -> str:
    """Texto con `n` oraciones de `palabras` palabras, separadas por punto."""
    return " ".join(" ".join([f"{prefijo}{i}"] * palabras) + "." for i in range(n))


def candidato(chunk_id: str, doc_id: str, texto: str) -> Candidato:
    return Candidato(
        chunk_id=chunk_id,
        doc_id=doc_id,
        texto=texto,
        fuente=f"{doc_id}.pdf",
        fenomeno=1,
        puntaje=1.0,
        posiciones={"bge-m3": 1},
    )


class TestSubdividir:
    def test_lo_corto_no_se_toca(self):
        texto = oraciones(2, 10)
        assert subdividir(texto) == [texto]

    def test_ninguna_pieza_excede_el_limite(self):
        # 20 oraciones de 20 palabras = 400 palabras, por encima de 250.
        piezas = subdividir(oraciones(20, 20))
        assert len(piezas) > 1
        assert all(len(p.split()) <= LIMITE for p in piezas)

    def test_no_se_pierde_texto(self):
        texto = oraciones(20, 20)
        recompuesto = " ".join(subdividir(texto))
        assert recompuesto.split() == texto.split()

    def test_una_oracion_gigante_se_corta_por_palabras(self):
        # Sin puntuación interna no hay frontera oracional a la que retroceder;
        # respetar el límite manda sobre conservar la oración entera.
        texto = " ".join(["x"] * 600)
        piezas = subdividir(texto)
        assert all(len(p.split()) <= LIMITE for p in piezas)
        assert sum(len(p.split()) for p in piezas) == 600


class TestConstruirFragmentos:
    def test_las_piezas_conservan_el_chunk_id_de_origen(self):
        # §9.2.1: al dividir, todas las piezas comparten el chunk_id original.
        largo = oraciones(20, 20)
        salida = construir_fragmentos(
            [candidato("c1", "D1", largo)], top=10, subdividir_largos=True
        )
        assert len(salida) > 1
        assert {f["chunk_id"] for f in salida} == {"c1"}

    def test_los_rangos_son_consecutivos_desde_uno(self):
        cands = [candidato(f"c{i}", f"D{i}", oraciones(20, 20)) for i in range(6)]
        salida = construir_fragmentos(cands, top=10)
        assert [f["rank"] for f in salida] == list(range(1, len(salida) + 1))

    def test_ningun_documento_acapara_el_top(self):
        # Tres fragmentos largos del mismo documento darían seis posiciones al
        # subdividir; la diversificación existe justamente para impedirlo.
        cands = [candidato(f"c{i}", "MISMO", oraciones(20, 20)) for i in range(3)]
        salida = construir_fragmentos(cands, top=10, subdividir_largos=True)
        assert len(salida) == config.MAX_FRAGMENTOS_POR_DOC

    def test_se_llenan_las_diez_posiciones(self):
        cands = [candidato(f"c{i}", f"D{i}", oraciones(20, 20)) for i in range(10)]
        salida = construir_fragmentos(cands, top=10)
        assert len(salida) == 10

    def test_por_defecto_se_recorta_y_no_se_subdivide(self):
        # Lo medido: recortar entrega 250 palabras en cada una de las diez
        # posiciones, mientras subdividir gasta rangos en colas de 71.
        largo = oraciones(20, 20)
        salida = construir_fragmentos([candidato("c1", "D1", largo)], top=10)
        assert len(salida) == 1
        assert len(salida[0]["text"].split()) <= LIMITE


class TestEsquema:
    def test_el_objeto_cumple_la_tabla_2(self):
        cands = [candidato(f"c{i}", f"D{i}", oraciones(20, 20)) for i in range(8)]
        objeto = resultado_a_dict(Resultado("q001", ["D0", "D1", "D2"], cands))

        assert objeto["query_id"] == "q001"
        assert len(objeto["documents"]) == config.TOP_DOCUMENTOS
        assert len(objeto["fragments"]) == config.TOP_FRAGMENTOS
        assert [d["rank"] for d in objeto["documents"]] == [1, 2, 3]
        assert [f["rank"] for f in objeto["fragments"]] == list(range(1, 11))
        for fragmento in objeto["fragments"]:
            assert set(fragmento) == {"rank", "chunk_id", "doc_id", "text"}
            assert len(fragmento["text"].split()) <= LIMITE

    def test_se_rellena_cuando_faltan_candidatos(self):
        # El esquema exige exactamente 3 y 10; una lista corta es descartada.
        uno = [candidato("c1", "D1", oraciones(1, 5))]
        objeto = resultado_a_dict(Resultado("q001", ["D1"], uno))
        assert len(objeto["documents"]) == 3
        assert len(objeto["fragments"]) == 10
        assert [f["rank"] for f in objeto["fragments"]] == list(range(1, 11))
