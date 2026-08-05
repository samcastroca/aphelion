"""La estrategia jerárquica: secciones como unidad, sin romper la fija.

Lo que estas pruebas protegen, en orden de gravedad si se rompe:

1. **Que ninguna oración se parta.** Es requisito del reto (§3.3) y la estrategia
   nueva subdivide secciones largas, que es justo donde se rompería.
2. **Que la caché no colisione.** Dos fragmentaciones del mismo tamaño y solape
   producen fragmentos distintos si una parte por secciones; si comparten clave,
   el barrido mide la fija creyendo medir la jerárquica.
3. **Que un documento sin estructura no se quede sin fragmentar.** JSON, CSV y
   los PDFs escaneados no declaran secciones y tienen que caer a la fija.
4. **Que la metadata obligatoria siga completa**, que es lo que evalúa el jurado.

    uv run pytest tests/test_chunking_jerarquico.py
"""

from __future__ import annotations

import pytest

from aphelion import config
from aphelion.indice import chunking
from aphelion.indice.estructura import Seccion

TOKENIZADOR = None


@pytest.fixture(scope="module")
def tok():
    return chunking._tokenizador(config.ENCODER_PRINCIPAL)


DOC = {
    "doc_id": "F1-PRUEBA-001",
    "fuente": "prueba.pdf",
    "formato": "pdf",
    "fenomeno": 1,
    "observatorio": "PRUEBA",
    "ruta_rel": "x/prueba.pdf",
    "meta": {"titulo": "Título del documento"},
}


def oracion(n: int) -> str:
    """Una oración con contenido suficiente para pesar varios tokens."""
    return (
        f"La observación número {n} describe un incidente registrado por el "
        f"sistema de monitoreo satelital durante el periodo analizado."
    )


class TestSinEstructura:
    def test_sin_secciones_delega_en_la_fija(self):
        """Un JSON o un PDF escaneado no puede quedarse sin fragmentar."""
        texto = " ".join(oracion(i) for i in range(30))
        jerarquico = chunking.fragmentar_jerarquico(DOC, texto, "es", None)
        fijo = chunking.fragmentar(DOC, texto, "es")
        assert [f.to_dict() for f in jerarquico] == [f.to_dict() for f in fijo]
        assert jerarquico, "delegar no puede significar devolver nada"

    def test_secciones_vacias_tras_limpiar_tambien_delegan(self):
        texto = " ".join(oracion(i) for i in range(10))
        vacias = [Seccion(titulo=None, texto="   ", nivel=0)]
        assert chunking.fragmentar_jerarquico(DOC, texto, "es", vacias)


class TestUnidadSeccion:
    def test_una_seccion_corta_se_entrega_entera(self, tok):
        """Por debajo del umbral no se subdivide, aunque sobre presupuesto."""
        corta = " ".join(oracion(i) for i in range(6))
        secciones = [
            Seccion("Introducción", corta, 1),
            Seccion("Metodología", corta, 1),
        ]
        frags = chunking.fragmentar_jerarquico(DOC, corta, "es", secciones)
        assert len(frags) == 2
        assert frags[0].titulo == "Introducción"
        assert frags[1].titulo == "Metodología"
        for f in frags:
            assert f.num_tokens < config.SECCION_MIN_TOKENS

    def test_una_seccion_larga_se_subdivide(self, tok):
        larga = " ".join(oracion(i) for i in range(120))
        assert chunking._contar(tok, larga) > config.SECCION_MAX_TOKENS
        secciones = [Seccion("Capítulo", larga, 1), Seccion("Otro", oracion(1) * 4, 1)]
        frags = chunking.fragmentar_jerarquico(DOC, larga, "es", secciones)
        del_capitulo = [f for f in frags if f.titulo == "Capítulo"]
        assert len(del_capitulo) > 1, "no se subdividió"
        for f in del_capitulo:
            assert f.num_tokens <= config.CHUNK_PRESUPUESTO

    def test_ningun_fragmento_supera_el_techo(self, tok):
        larga = " ".join(oracion(i) for i in range(200))
        secciones = [Seccion("A", larga, 1), Seccion("B", larga, 1)]
        frags = chunking.fragmentar_jerarquico(DOC, larga, "es", secciones)
        for f in frags:
            assert f.num_tokens <= config.SECCION_MAX_TOKENS

    def test_las_secciones_minusculas_se_fusionan(self, tok):
        """Cuarenta epígrafes de dos líneas no pueden ser cuarenta fragmentos."""
        secciones = [Seccion(f"Epígrafe {i}", oracion(i), 2) for i in range(40)]
        frags = chunking.fragmentar_jerarquico(DOC, "x", "es", secciones)
        assert len(frags) < 40
        for f in frags:
            assert f.num_tokens >= config.MIN_TOKENS_FRAGMENTO


class TestCompletitudLinguistica:
    def test_ninguna_oracion_queda_partida(self, tok):
        """§3.3: la frontera cae entre oraciones, también al subdividir."""
        oraciones = [oracion(i) for i in range(150)]
        larga = " ".join(oraciones)
        secciones = [Seccion("Larga", larga, 1), Seccion("Corta", oracion(999), 1)]
        frags = chunking.fragmentar_jerarquico(DOC, larga, "es", secciones)

        # Cada oración original tiene que aparecer entera en algún fragmento.
        unido = " || ".join(f.texto for f in frags)
        faltan = [o for o in oraciones if o not in unido]
        assert not faltan, f"{len(faltan)} oraciones quedaron partidas"


class TestFronteraAMitadDeFrase:
    """El caso real que apareció midiendo F1-CSET-005.

    La heurística tomó por epígrafe una línea de texto corrido en negrita, así
    que la frontera de sección cayó a mitad de una oración y esa oración quedó
    repartida entre dos fragmentos. Detectar mejor reduce el caso; realinear a
    frontera de oración lo elimina.
    """

    def test_la_oracion_partida_se_recompone(self):
        # La primera sección corta a media frase; la segunda la continúa.
        secciones = [
            Seccion("Intro", "Este informe analiza un enfoque", 1),
            Seccion("analitico para orientar", "analitico para orientar la "
                    "decision de politica publica en el periodo estudiado. "
                    + " ".join(oracion(i) for i in range(4)), 1),
        ]
        frags = chunking.fragmentar_jerarquico(DOC, "x", "es", secciones)
        unido = " || ".join(f.texto for f in frags)
        assert "Este informe analiza un enfoque analitico para orientar" in unido, (
            "la oración sigue partida entre dos fragmentos"
        )

    def test_una_seccion_que_cierra_frase_no_se_toca(self):
        completa = " ".join(oracion(i) for i in range(5))
        secciones = [
            Seccion("Uno", completa, 1),
            Seccion("Dos", completa, 1),
        ]
        frags = chunking.fragmentar_jerarquico(DOC, "x", "es", secciones)
        # Dos secciones que cierran frase siguen siendo dos fragmentos, cada uno
        # con su título: nada se movió de sitio.
        assert [f.titulo for f in frags] == ["Uno", "Dos"]
        assert frags[0].texto == completa

    def test_la_negrita_a_media_frase_no_abre_seccion(self):
        """En el detector: si lo anterior no cerró, esto es su continuación."""
        from aphelion.indice import estructura

        cuerpo = 10.0
        negrita = {"texto": "un termino destacado", "tam": 10.0, "negrita": True}
        tras_frase = {"texto": "Termina la idea anterior.", "tam": 10.0,
                      "negrita": False}
        a_medias = {"texto": "la frase sigue y no ha cerrado", "tam": 10.0,
                    "negrita": False}
        assert estructura._es_encabezado(negrita, cuerpo, tras_frase)
        assert not estructura._es_encabezado(negrita, cuerpo, a_medias)

    def test_un_salto_de_tamano_si_abre_seccion_aunque_venga_a_medias(self):
        """Nadie compone media frase en cuerpo 20 dentro de un párrafo en 10."""
        from aphelion.indice import estructura

        titulo = {"texto": "3. Resultados", "tam": 20.0, "negrita": False}
        a_medias = {"texto": "la frase sigue y no ha cerrado", "tam": 10.0,
                    "negrita": False}
        assert estructura._es_encabezado(titulo, 10.0, a_medias)


class TestMetadata:
    def test_conserva_los_campos_obligatorios(self):
        secciones = [
            Seccion("Uno", " ".join(oracion(i) for i in range(5)), 1),
            Seccion("Dos", " ".join(oracion(i) for i in range(5)), 1),
        ]
        frag = chunking.fragmentar_jerarquico(DOC, "x", "es", secciones)[0].to_dict()
        for campo in ("doc_id", "chunk_id", "fuente", "formato", "fenomeno",
                      "posicion", "num_tokens", "texto"):
            assert campo in frag, f"falta {campo}"
        assert frag["doc_id"] == "F1-PRUEBA-001"
        assert frag["chunk_id"] == "F1-PRUEBA-001-chunk-0000"
        assert frag["posicion"] == 0

    def test_el_titulo_es_el_de_la_seccion_no_el_del_documento(self):
        secciones = [
            Seccion("Sección propia", " ".join(oracion(i) for i in range(5)), 1),
            Seccion("Otra", " ".join(oracion(i) for i in range(5)), 1),
        ]
        frags = chunking.fragmentar_jerarquico(DOC, "x", "es", secciones)
        assert frags[0].titulo == "Sección propia"

    def test_las_posiciones_son_consecutivas_desde_cero(self):
        secciones = [
            Seccion(f"S{i}", " ".join(oracion(j) for j in range(5)), 1)
            for i in range(4)
        ]
        frags = chunking.fragmentar_jerarquico(DOC, "x", "es", secciones)
        assert [f.posicion for f in frags] == list(range(len(frags)))


class TestClaveDeCache:
    def test_la_estrategia_fija_conserva_su_clave(self):
        """Renombrarla invalidaría todos los índices ya construidos."""
        assert config.clave_chunking(504, 0.15) == "c504-s015"
        assert config.clave_chunking(504, 0.15, config.ESTRATEGIA_FIJA) == "c504-s015"

    def test_la_jerarquica_no_colisiona_con_la_fija(self):
        fija = config.clave_chunking(504, 0.15, config.ESTRATEGIA_FIJA)
        jer = config.clave_chunking(504, 0.15, config.ESTRATEGIA_JERARQUICA)
        assert fija != jer

    def test_las_rutas_tampoco(self):
        assert (config.ruta_fragmentos_prueba(504, 0.15)
                != config.ruta_fragmentos_prueba(504, 0.15,
                                                 config.ESTRATEGIA_JERARQUICA))
        assert (config.ruta_indices_prueba(504, 0.15)
                != config.ruta_indices_prueba(504, 0.15,
                                              config.ESTRATEGIA_JERARQUICA))


class TestLimpiezaCompartida:
    """Las dos estrategias tienen que recibir el mismo texto limpio.

    Si no, la comparación mide el preprocesado: la jerárquica trocea antes de
    limpiar, y una cabecera que se repite entre páginas no se repite dentro de
    una sección, así que sobreviviría solo en esa rama.
    """

    def test_quitar_boilerplate_no_cambio_de_comportamiento(self):
        from aphelion.ingesta import limpieza

        texto = "\n".join(
            f"CABECERA\nlinea util {i}" for i in range(5)
        )
        assert "CABECERA" not in limpieza.quitar_boilerplate(texto)
        assert "linea util 1" in limpieza.quitar_boilerplate(texto)

    def test_el_boilerplate_del_documento_se_aplica_a_cada_seccion(self):
        from aphelion.ingesta import limpieza

        documento = "\n".join(
            f"CABECERA INSTITUCIONAL\ncontenido de la pagina {i}" for i in range(6)
        )
        basura = limpieza.lineas_boilerplate(limpieza.normalizar(documento))
        assert "CABECERA INSTITUCIONAL" in basura

        # Una sección suelta no tiene repeticiones suficientes para reconocerla,
        # y por eso el conjunto se calcula fuera y se pasa.
        seccion = "CABECERA INSTITUCIONAL\ncontenido de la pagina 3"
        assert "CABECERA" in limpieza.limpiar_seccion(seccion, set())
        assert "CABECERA" not in limpieza.limpiar_seccion(seccion, basura)
        assert "contenido de la pagina 3" in limpieza.limpiar_seccion(seccion, basura)


class TestRecetaJerarquica:
    def test_esta_en_el_catalogo_y_solo_cambia_el_chunking(self):
        from aphelion.evaluacion.recetas import RECETAS

        entrega, jer = RECETAS["entrega"], RECETAS["jerarquico-v1"]
        assert jer.estrategia == config.ESTRATEGIA_JERARQUICA
        assert entrega.estrategia == config.ESTRATEGIA_FIJA
        # Todo lo demás idéntico: si cambiara algo más, la diferencia medida
        # admitiría dos explicaciones.
        assert jer.encoders == entrega.encoders
        assert jer.politica() == entrega.politica()
        assert jer.solape == entrega.solape

    def test_no_tiene_problemas_con_bge(self):
        from aphelion.evaluacion.recetas import RECETAS

        assert RECETAS["jerarquico-v1"].problemas() == []

    def test_el_techo_de_la_jerarquica_descarta_encoders_de_ventana_corta(self):
        """Una sección de 650 tokens no cabe en mE5 y se truncaría en silencio."""
        from dataclasses import replace
        from aphelion.evaluacion.recetas import RECETAS

        con_me5 = replace(RECETAS["jerarquico-v1"], encoders=("me5-large",))
        assert con_me5.techo_tokens == config.SECCION_MAX_TOKENS
        assert any("truncaría" in f for f in con_me5.problemas())
