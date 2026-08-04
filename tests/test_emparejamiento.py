"""Comprueba el emparejamiento de fragmentos por texto.

Es la pieza que hace válido el barrido de chunking: sin ella, probar 256 tokens
contra un ground truth anotado sobre 504 daría ceros en vez de medidas. Y como
además es la clave con la que empareja el jurado (§10.2.1), equivocarse aquí
significa optimizar contra una métrica que no es la que se va a puntuar.

    uv run pytest
"""

from __future__ import annotations

import pytest

from aphelion.evaluacion.emparejamiento import (
    Emparejador,
    JuicioTextual,
    ngramas,
    normalizar,
    solape,
)


def juicio(texto: str, relevancia: float) -> JuicioTextual:
    return JuicioTextual(
        chunk_id="D-chunk-0000",
        doc_id="D",
        relevancia=relevancia,
        bolsa=ngramas(texto),
    )


FRASES = [
    "La congestion de la orbita baja terrestre incrementa el riesgo de colisiones.",
    "Los satelites obsoletos y los fragmentos de cohetes son desechos orbitales.",
    "Las pruebas antisatelite destructivas generan miles de fragmentos rastreables.",
    "El Tratado del Espacio Ultraterrestre prohibe las armas nucleares en orbita.",
    "La guerra electronica interfiere las senales de navegacion por satelite.",
    "Las maniobras de proximidad permiten inspeccionar satelites extranjeros.",
]


def parrafo(desde: int, hasta: int) -> str:
    return " ".join(FRASES[desde:hasta])


class TestNormalizar:
    def test_quita_tildes_y_puntuacion(self):
        assert normalizar("Órbita, baja!") == "orbita baja"

    def test_colapsa_espacios(self):
        assert normalizar("a   b\n\nc") == "a b c"


class TestSolape:
    def test_identico_es_uno(self):
        b = ngramas(parrafo(0, 2))
        assert solape(b, b) == pytest.approx(1.0)

    def test_disjunto_es_cero(self):
        assert solape(ngramas(parrafo(0, 1)), ngramas(parrafo(4, 5))) == 0.0

    def test_bolsa_vacia_no_revienta(self):
        assert solape(frozenset(), ngramas(parrafo(0, 1))) == 0.0

    def test_contencion_en_los_dos_sentidos(self):
        # Es la propiedad que hace válido el barrido de chunking: el fragmento
        # devuelto puede ser más grande que el juzgado (chunks de 768) o más
        # pequeño (chunks de 256), y los dos casos deben emparejar.
        pequeno = ngramas(parrafo(0, 1))
        grande = ngramas(parrafo(0, 4))
        assert solape(pequeno, grande) == pytest.approx(1.0)
        assert solape(grande, pequeno) == pytest.approx(1.0)


class TestEmparejador:
    def test_el_texto_identico_hereda_su_grado(self):
        e = Emparejador([juicio(parrafo(0, 2), 2.0)])
        assert e.relevancia(parrafo(0, 2)) == 2.0

    def test_un_chunk_mas_grande_que_contiene_al_juzgado_hereda(self):
        # Chunks de 768 tokens: el devuelto envuelve al juzgado. Si contiene el
        # pasaje que responde, responde.
        e = Emparejador([juicio(parrafo(0, 2), 2.0)])
        assert e.relevancia(parrafo(0, 4)) == 2.0

    def test_un_chunk_mas_pequeno_dentro_del_juzgado_hereda(self):
        # Chunks de 256 tokens: el devuelto cae dentro del juzgado.
        e = Emparejador([juicio(parrafo(0, 4), 2.0)])
        assert e.relevancia(parrafo(0, 2)) == 2.0

    def test_texto_ajeno_no_hereda_nada(self):
        e = Emparejador([juicio(parrafo(0, 2), 2.0)])
        assert e.relevancia(parrafo(4, 6)) == 0.0

    def test_toma_el_grado_maximo_entre_los_que_solapan(self):
        e = Emparejador([juicio(parrafo(0, 2), 1.0), juicio(parrafo(2, 4), 2.0)])
        assert e.relevancia(parrafo(0, 4)) == 2.0

    def test_un_solape_por_debajo_del_umbral_no_cuenta(self):
        # Una sola frase compartida de seis no debería arrastrar el grado: si
        # bastara, cualquier fragmento del mismo informe heredaría relevancia.
        ruido = " ".join(f"palabra{i} ajena{i} distinta{i}" for i in range(40))
        e = Emparejador([juicio(parrafo(0, 6), 2.0)])
        assert e.relevancia(FRASES[0] + " " + ruido) == 0.0

    def test_el_relleno_repetido_diluye_menos_que_el_variado(self):
        # Propiedad del emparejamiento por conjuntos que conviene tener fijada:
        # los n-gramas se deduplican, así que un relleno que se repite ocupa
        # mucho texto pero poca bolsa y diluye menos que un relleno variado de la
        # misma longitud. En este corpus no llega a cruzar el umbral —lo
        # comprueba la prueba anterior— y `limpieza.quitar_boilerplate` elimina
        # justamente las líneas repetidas antes de fragmentar. Queda fijado para
        # que se note si algún día ese margen se estrecha.
        juzgado = ngramas(parrafo(0, 6))
        repetido = ngramas(FRASES[0] + " " + " ".join(["relleno repetido igual"] * 40))
        variado = ngramas(
            FRASES[0] + " " + " ".join(f"relleno{i} variado{i} distinto{i}" for i in range(40))
        )
        assert solape(repetido, juzgado) > solape(variado, juzgado)

    def test_el_ideal_son_las_relevancias_anotadas(self):
        e = Emparejador([juicio(parrafo(0, 1), 2.0), juicio(parrafo(1, 2), 1.0)])
        assert sorted(e.ideal, reverse=True) == [2.0, 1.0]

    def test_sin_juicios_todo_es_cero(self):
        e = Emparejador([])
        assert e.relevancias([parrafo(0, 2), parrafo(2, 4)]) == [0.0, 0.0]

    def test_un_fragmento_de_solo_titulo_no_revienta(self):
        # Los artículos del CEEEP entraron al índice con un título de 15 tokens,
        # por debajo de la ventana de n-gramas.
        corto = "Inteligencia Artificial y Ciberdefensa"
        e = Emparejador([juicio(corto, 1.0)])
        assert e.relevancia(corto) == 1.0
        assert e.relevancia("otro titulo por completo") == 0.0
