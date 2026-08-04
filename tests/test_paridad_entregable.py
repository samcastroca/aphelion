"""Paridad paquete ↔ entregable sobre un índice sintético, sin cargar modelos.

`scripts/etapas/06_verificar.py` ya compara las dos implementaciones, pero exige
el índice real: horas de GPU antes de descubrir una divergencia. Esta prueba
hace la misma comparación en milisegundos, con un índice falso y encoders
falsos, de modo que un cambio que toque una sola de las dos copias de la
política de recuperación falle aquí, en el primer `pytest`.

No sustituye a 06_verificar —que además prueba la carga real de FAISS y la
lectura del PDF de consultas—, lo adelanta.

    uv run pytest
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aphelion.busqueda import salida
from aphelion.busqueda.consultas import Consulta
from aphelion.busqueda.recuperacion import Recuperador
from aphelion.indice.vectores import IndiceVectorial

RAIZ = Path(__file__).resolve().parents[1]


def cargar_generador():
    ruta = RAIZ / "entrega" / "generador.py"
    nombre = "generador_paridad"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


class FalsoFaiss:
    """Devuelve un ranking fijo, con la interfaz de search() de FAISS."""

    def __init__(self, orden: list[int], puntajes: list[float]):
        self.orden = np.asarray(orden, dtype=np.int64)
        self.puntajes = np.asarray(puntajes, dtype=np.float32)

    def search(self, vector, k):
        n = min(k, len(self.orden))
        return self.puntajes[:n].reshape(1, -1), self.orden[:n].reshape(1, -1)


class FalsoEncoder:
    """Vale para los dos lados: el vector no importa, FalsoFaiss lo ignora."""

    def codificar_consultas(self, textos):
        return np.zeros((len(textos), 4), dtype=np.float32)

    def encode(self, texts):
        return np.zeros((len(texts), 4), dtype=np.float32)


def _texto(i: int) -> str:
    if i < 2:
        # Largo: 20 oraciones de 15 palabras = 300, por encima del límite de 250.
        return " ".join(
            " ".join([f"c{i}s{j}w{k}" for k in range(14)]) + " fin." for j in range(20)
        )
    if i == 5 or i == 13:
        # Texto idéntico en dos documentos distintos: ejercita la deduplicación
        # sin volver inalcanzable a ninguno de los dos.
        return "Texto repetido en dos informes distintos. Aparece dos veces."
    return " ".join(
        " ".join([f"c{i}s{j}w{k}" for k in range(9)]) + " fin." for j in range(3)
    )


def _metadata(n: int = 24) -> list[dict]:
    filas = []
    for i in range(n):
        doc = f"D{i % 8}"
        filas.append(
            {
                "chunk_id": f"{doc}-chunk-{i:04d}",
                "doc_id": doc,
                # Dos doc_id por fuente: ejercita la agregación por fuente.
                "fuente": f"fuente{(i % 8) // 2}.pdf",
                "formato": "pdf",
                "fenomeno": (i % 3) + 1,
                "posicion": i,
                "num_tokens": 100,
                "texto": _texto(i),
                "idioma": ("es", "en", "pt")[i % 3],
            }
        )
    return filas


def _rankings(n: int = 24):
    # Dos órdenes distintos para que la fusión tenga algo que fusionar.
    orden_a = list(range(n))
    orden_b = [(i * 7) % n for i in range(n)]  # permutación (7 y 24 son coprimos)
    puntajes_a = [0.90 - 0.01 * i for i in range(n)]
    puntajes_b = [0.80 - 0.005 * i for i in range(n)]
    return (orden_a, puntajes_a), (orden_b, puntajes_b)


@pytest.fixture(scope="module")
def gen():
    return cargar_generador()


def _lado_paquete(umbral: float | None) -> dict:
    meta = _metadata()
    (oa, pa), (ob, pb) = _rankings()
    indices = {
        "bge-m3": IndiceVectorial("bge-m3", FalsoFaiss(oa, pa), meta),
        "me5-large": IndiceVectorial("me5-large", FalsoFaiss(ob, pb), meta),
    }
    falso = FalsoEncoder()
    recuperador = Recuperador(
        indices,
        encoders_cargados={"bge-m3": falso, "me5-large": falso},
        umbral_relativo=umbral,
    )
    resultado = recuperador.recuperar(Consulta("q001", "consulta de prueba"))
    return salida.resultado_a_dict(resultado)


def _lado_entregable(gen, umbral: float | None) -> dict:
    meta = _metadata()
    (oa, pa), (ob, pb) = _rankings()
    indices = {
        "bge-m3": gen.VectorIndex("bge-m3", FalsoFaiss(oa, pa), meta),
        "me5-large": gen.VectorIndex("me5-large", FalsoFaiss(ob, pb), meta),
    }
    retriever = gen.Retriever(indices, threshold=umbral)
    falso = FalsoEncoder()
    retriever._encoders = {"bge-m3": falso, "me5-large": falso}
    documentos, fragmentos = retriever.retrieve(gen.Query("q001", "consulta de prueba"))
    return gen.build_record("q001", documentos, fragmentos)


class TestParidad:
    @pytest.mark.parametrize("umbral", [None, 0.9])
    def test_mismo_indice_misma_salida(self, gen, umbral):
        assert _lado_paquete(umbral) == _lado_entregable(gen, umbral)

    def test_la_salida_cumple_el_esquema(self, gen, tmp_path):
        objeto = _lado_entregable(gen, None)
        destino = tmp_path / "resultados.jsonl"
        destino.write_text(
            json.dumps(objeto, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        assert gen.validate(destino, expected_queries=1) == []

    def test_el_umbral_recorta_el_pool(self, gen):
        # Con umbral 0.9 el índice A corta en 0.81 y el B en 0.72: el pool
        # fusionado es más pequeño, pero el esquema sigue completo.
        objeto = _lado_entregable(gen, 0.9)
        assert len(objeto["documents"]) == 3
        assert len(objeto["fragments"]) == 10


class TestAgregacionEnElEntregable:
    """El generador tenía una sola forma de pasar de fragmentos a documentos.

    `max` era la única implementada, y es la que se entrega. Pero el barrido
    mide `top2` y `top3` como alternativas, así que sin ellas en el generador
    una mejora medida en el paquete no se puede entregar: el jurado ejecuta
    este archivo, no `aphelion`.

    Lo que estos tests fijan es que las dos copias de la política den lo mismo,
    que es de lo que trata este módulo entero.
    """

    def candidatos(self, gen):
        """Un documento con evidencia repartida contra otro con un solo pico.

        Con `max` gana D2 (0.90 contra 0.80); con `top2` gana D1, porque sus dos
        mejores promedian 0.775 contra los 0.50 de D2. Es el caso que distingue
        las dos agregaciones, y por eso vale para comprobar que las dos copias
        coinciden.
        """
        return [
            gen.Candidate(chunk_id="c1", doc_id="D1", text="t1", fuente="D1.pdf",
                          fenomeno=1, score=0.80, ranks={"e": 1}, idioma="es"),
            gen.Candidate(chunk_id="c2", doc_id="D1", text="t2", fuente="D1.pdf",
                          fenomeno=1, score=0.75, ranks={"e": 2}, idioma="es"),
            gen.Candidate(chunk_id="c3", doc_id="D2", text="t3", fuente="D2.pdf",
                          fenomeno=1, score=0.90, ranks={"e": 3}, idioma="es"),
            gen.Candidate(chunk_id="c4", doc_id="D2", text="t4", fuente="D2.pdf",
                          fenomeno=1, score=0.50, ranks={"e": 4}, idioma="es"),
        ]

    def test_max_sigue_siendo_el_comportamiento_por_defecto(self):
        """Cambiar el defecto cambiaría lo que se entrega sin que nadie lo pida."""
        gen = cargar_generador()
        assert gen.aggregate_to_documents(self.candidatos(gen))[0] == "D2"

    def test_top2_premia_la_evidencia_repartida(self):
        gen = cargar_generador()
        docs = gen.aggregate_to_documents(self.candidatos(gen), mode="top2")
        assert docs[0] == "D1"

    def test_el_generador_agrega_igual_que_el_paquete(self):
        from aphelion.busqueda.recuperacion import agregar_a_documentos, Candidato

        gen = cargar_generador()
        propios = [
            Candidato(chunk_id=c.chunk_id, doc_id=c.doc_id, texto=c.text,
                      fuente=c.fuente, fenomeno=c.fenomeno, puntaje=c.score,
                      posiciones=dict(c.ranks), idioma=c.idioma)
            for c in self.candidatos(gen)
        ]
        for modo in ("max", "top2", "top3"):
            assert (gen.aggregate_to_documents(self.candidatos(gen), mode=modo)
                    == agregar_a_documentos(propios, modo=modo)), modo

    def test_un_modo_desconocido_no_pasa_por_max_en_silencio(self):
        """Un modo mal escrito en la línea de comandos tiene que fallar, no
        entregar otra cosa sin avisar."""
        gen = cargar_generador()
        with pytest.raises(ValueError):
            gen.aggregate_to_documents(self.candidatos(gen), mode="topdos")
