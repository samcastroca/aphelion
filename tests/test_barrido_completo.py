"""Comprueba la rejilla del barrido, las recetas y las estrategias que compara.

Tres cosas que, si están mal, hacen que el barrido elija la configuración
equivocada sin que nada avise: la deduplicación de la rejilla (que evita medir
cientos de veces lo mismo y falsear el Conteo de Borda), las recetas —donde una
receta incompleta o que no cabe en la ventana del encoder mide algo distinto de
lo que declara— y las agregaciones y fusiones nuevas, cuyo comportamiento hay que
fijar antes de decidir con ellas.

    uv run pytest
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from aphelion import config
from aphelion.busqueda.recuperacion import (
    Candidato,
    _puntaje_documento,
    agregar_a_documentos,
    fusionar_convexa,
    fusionar_rrf,
)
from aphelion.evaluacion import recetas as mod_recetas

RAIZ = Path(__file__).resolve().parents[1]


def cargar_barrido():
    ruta = RAIZ / "scripts" / "analisis" / "barrido_completo.py"
    spec = importlib.util.spec_from_file_location("barrido_completo", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["barrido_completo"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def meta(chunk_id: str, doc_id: str = "D1") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "texto": f"texto de {chunk_id}",
        "fuente": f"{doc_id}.pdf",
        "fenomeno": 1,
        "idioma": "es",
    }


def candidato(chunk_id: str, doc_id: str, puntaje: float) -> Candidato:
    return Candidato(
        chunk_id=chunk_id,
        doc_id=doc_id,
        texto=f"texto de {chunk_id}",
        fuente=f"{doc_id}.pdf",
        fenomeno=1,
        puntaje=puntaje,
        posiciones={"bge-m3": 1},
    )


def opciones(mod, **cambios):
    """Las opciones por defecto con lo que se le pase encima."""
    op = dict(mod.DEFECTOS)
    op.update(cambios)
    op.setdefault("max_fusion", 2)
    return op


class TestPoliticas:
    def test_k0_no_multiplica_las_fusiones_que_no_lo_usan(self):
        # k0 solo interviene en RRF. Sin deduplicar, combsum y convexa
        # aparecerían tres veces idénticas y Borda las premiaría por repetición.
        mod = cargar_barrido()
        op = opciones(
            mod,
            fusiones=["rrf", "combsum", "convexa"],
            k0=[10, 20, 60],
            boosts=[1.0],
            max_por_doc=[3],
            candidatos=[200],
            umbrales=[None],
            subdividir=[False],
            agregaciones=["max"],
        )
        combos = mod.politicas(op)
        assert len(combos) == 5  # 3 de rrf + 1 combsum + 1 convexa
        assert sum(1 for c in combos if c["fusion"] == "combsum") == 1

    def test_elegir_una_sola_opcion_da_una_sola_politica(self):
        # Es el punto de que las dimensiones se elijan: pedir una cosa de cada
        # una tiene que dar exactamente una corrida, no un producto cartesiano.
        mod = cargar_barrido()
        op = opciones(
            mod,
            fusiones=["rrf"], k0=[60], boosts=[1.0], max_por_doc=[3],
            candidatos=[200], umbrales=[None], subdividir=[False],
            agregaciones=["max"],
        )
        assert len(mod.politicas(op)) == 1

    def test_el_defecto_es_pequeno(self):
        # Correr el guion sin pedir nada no debe lanzar una noche de GPU.
        mod = cargar_barrido()
        assert len(mod.politicas(opciones(mod))) <= 8

    def test_cada_dimension_multiplica_lo_que_se_le_pide(self):
        mod = cargar_barrido()
        base = len(mod.politicas(opciones(mod, agregaciones=["max"])))
        doble = len(mod.politicas(opciones(mod, agregaciones=["max", "top2"])))
        assert doble == 2 * base


class TestPresets:
    def test_todos_los_presets_son_utilizables(self):
        mod = cargar_barrido()
        for nombre, preset in mod.PRESETS.items():
            op = dict(mod.DEFECTOS)
            op.update(preset)
            op["max_fusion"] = 2
            assert mod.politicas(op), f"el preset {nombre} no produce políticas"
            assert op["encoders"], f"el preset {nombre} se queda sin encoders"

    def test_los_presets_solo_tocan_dimensiones_que_existen(self):
        # Una clave mal escrita en un preset se ignoraría en silencio y el
        # experimento correría con el valor por defecto sin avisar.
        mod = cargar_barrido()
        for nombre, preset in mod.PRESETS.items():
            desconocidas = set(preset) - set(mod.DEFECTOS)
            assert not desconocidas, f"{nombre} usa claves inexistentes: {desconocidas}"

    def test_rapido_es_el_mas_barato(self):
        mod = cargar_barrido()
        op_rapido = dict(mod.DEFECTOS); op_rapido.update(mod.PRESETS["rapido"])
        op_todo = dict(mod.DEFECTOS); op_todo.update(mod.PRESETS["todo"])
        assert len(mod.politicas(op_rapido)) < len(mod.politicas(op_todo))
        assert len(op_rapido["encoders"]) < len(op_todo["encoders"])

    def test_los_encoders_de_los_presets_estan_en_el_catalogo(self):
        mod = cargar_barrido()
        for nombre, preset in mod.PRESETS.items():
            for encoder in preset.get("encoders", []):
                assert encoder in config.ENCODERS, f"{nombre}: {encoder} no existe"

    def test_el_preset_de_chunking_cabe_entero_en_su_encoder(self):
        # Es el preset que se recomienda correr primero, y su conclusión es un
        # orden entre chunkings: si uno de ellos se codificara truncado, ese orden
        # sería falso justo en el extremo que interesa.
        mod = cargar_barrido()
        preset = mod.PRESETS["chunking"]
        for encoder in preset["encoders"]:
            for chunk in preset["chunks"]:
                assert config.cabe_en_ventana(chunk, encoder), f"{chunk} en {encoder}"

    def test_no_se_indexa_un_par_que_no_cabe_en_la_ventana(self):
        # `todo` sí mezcla chunkings con encoders que no los admiten: se descartan
        # en vez de construir un índice truncado que daría métricas plausibles.
        mod = cargar_barrido()
        op = dict(mod.DEFECTOS)
        op.update(mod.PRESETS["todo"])
        op["max_fusion"] = 1
        pares = mod.pares_a_indexar(mod.planificar_rejilla(op))
        for (chunk, _), encoders in pares.items():
            for encoder in encoders:
                assert config.cabe_en_ventana(chunk, encoder), f"{chunk} en {encoder}"
        assert "me5-small" not in pares.get((768, 0.15), [])
        assert "bge-m3" in pares[(768, 0.15)]


class TestRecetas:
    def test_ninguna_receta_declara_algo_que_no_puede_medir(self):
        # `problemas()` cubre las dos formas de mentir sin fallar: un encoder que
        # no existe, y un chunk más largo que su ventana, que se truncaría en
        # silencio y daría métricas plausibles de una configuración imaginaria.
        for nombre, receta in mod_recetas.RECETAS.items():
            assert receta.problemas() == [], f"{nombre}: {receta.problemas()}"

    def test_la_receta_entrega_es_de_verdad_la_entrega(self):
        # Es la vara de medir de todas las demás: si se desincroniza de config,
        # cada comparación se hace contra algo que no se entrega.
        r = mod_recetas.RECETAS["entrega"]
        assert r.encoders == tuple(config.ENCODERS_ENTREGA)
        assert r.chunk == config.CHUNK_PRESUPUESTO
        assert r.solape == config.CHUNK_SOLAPE
        assert r.k0 == config.RRF_K0
        assert r.boost == config.BOOST_FENOMENO
        assert r.max_por_doc == config.MAX_FRAGMENTOS_POR_DOC
        assert r.candidatos == config.CANDIDATOS_POR_INDICE
        assert r.umbral == config.UMBRAL_RELATIVO
        assert r.subdividir == config.SUBDIVIDIR_FRAGMENTOS
        assert (r.fusion, r.agregacion) == ("rrf", "max")

    def test_cada_receta_se_distingue_de_las_demas(self):
        # Dos recetas idénticas gastarían el doble de GPU en la misma medida y
        # Borda las premiaría por aparecer repetidas.
        vistas = {}
        for nombre, receta in mod_recetas.RECETAS.items():
            firma = (receta.encoders, receta.chunk, receta.solape,
                     tuple(sorted(receta.politica().items())))
            assert firma not in vistas, f"{nombre} repite a {vistas.get(firma)}"
            vistas[firma] = nombre

    def test_la_politica_de_una_receta_encaja_con_la_de_la_rejilla(self):
        # Las dos rutas de planificación tienen que producir la misma forma de
        # configuración, porque de ahí para abajo el barrido no las distingue.
        mod = cargar_barrido()
        de_rejilla = set(mod.politicas(opciones(mod))[0])
        for nombre, receta in mod_recetas.RECETAS.items():
            assert set(receta.politica()) == de_rejilla, nombre

    def test_los_nombres_viajan_por_la_linea_de_comandos_y_son_carpetas(self):
        for nombre in mod_recetas.RECETAS:
            assert nombre == nombre.strip()
            assert not set(nombre) & set(' ,\\/:*?"<>|'), nombre

    def test_seleccionar_respeta_el_orden_del_catalogo(self):
        # El catálogo está ordenado para que lo que comparte fragmentación vaya
        # primero: una corrida interrumpida deja hecho lo que se reutiliza.
        pedido = mod_recetas.seleccionar("contexto,entrega")
        assert pedido == ["entrega", "contexto"]

    def test_seleccionar_acepta_todas_y_rechaza_lo_que_no_existe(self):
        assert mod_recetas.seleccionar("todas") == list(mod_recetas.RECETAS)
        with pytest.raises(ValueError, match="inventada"):
            mod_recetas.seleccionar("entrega,inventada")

    def test_una_receta_es_una_corrida(self):
        # Es el punto de que existan: la tabla se lee sin descontar el
        # sobreajuste de haber probado miles de variantes.
        mod = cargar_barrido()
        elegidas = mod_recetas.seleccionar("todas")
        planes = mod.planificar_recetas(elegidas)
        assert len(planes) == len(elegidas)
        assert [p.receta for p in planes] == elegidas

    def test_el_plan_lleva_los_parametros_que_la_receta_declara(self):
        mod = cargar_barrido()
        plan = mod.planificar_recetas(["contexto"])[0]
        receta = mod_recetas.RECETAS["contexto"]
        assert plan.encoders == receta.encoders
        assert (plan.chunk, plan.solape) == (receta.chunk, receta.solape)
        assert plan.cfg == receta.politica()

    def test_no_indexa_un_encoder_en_un_chunking_que_nadie_pidio(self):
        # `contexto` pide 768 tokens con BGE-M3 y nada más. Indexar ahí mE5 sería
        # una hora de GPU en un índice truncado que ningún plan va a consultar.
        mod = cargar_barrido()
        pares = mod.pares_a_indexar(mod.planificar_recetas(["entrega", "contexto"]))
        assert pares[(768, 0.15)] == ["bge-m3"]
        assert set(pares[(config.CHUNK_PRESUPUESTO, config.CHUNK_SOLAPE)]) == {
            "bge-m3", "me5-large"
        }

    def test_el_resumen_no_nombra_una_fusion_que_no_interviene(self):
        # Con un solo índice no hay nada que fusionar; poner 'rrf' ahí haría
        # creer que se está midiendo algo que no participa.
        assert "sin fusión" in mod_recetas.RECETAS["bge-top2"].resumen()
        assert "rrf" in mod_recetas.RECETAS["entrega"].resumen()

    def test_el_resumen_enseña_lo_que_se_aparta_de_la_entrega(self):
        assert "umbral=0.9" in mod_recetas.RECETAS["filtrado"].resumen()
        assert "umbral" not in mod_recetas.RECETAS["entrega"].resumen()

    def test_toda_receta_explica_en_que_se_basa(self):
        # Sin el motivo, el catálogo es una lista de corazonadas y en seis meses
        # nadie sabe por qué se probó eso.
        for nombre, receta in mod_recetas.RECETAS.items():
            assert len(receta.apuesta) > 10, nombre
            assert len(receta.por_que) > 80, nombre


class TestRejilla:
    def test_los_subconjuntos_incluyen_los_solos_y_los_pares(self):
        mod = cargar_barrido()
        combos = mod.subconjuntos(["a", "b", "c"], 2)
        assert ("a",) in combos and ("b",) in combos and ("c",) in combos
        assert ("a", "b") in combos
        assert ("a", "b", "c") not in combos
        assert len(combos) == 6

    def test_la_clave_de_chunking_distingue_solapes(self):
        mod = cargar_barrido()
        assert mod.clave_chunking(504, 0.0) != mod.clave_chunking(504, 0.15)


class TestAgregacion:
    def test_max_toma_el_mejor_fragmento(self):
        assert _puntaje_documento([0.2, 0.9, 0.3], "max") == pytest.approx(0.9)

    def test_suma_tiene_sesgo_de_longitud(self):
        # Es exactamente el sesgo que el diseño descartó por argumento: cuarenta
        # fragmentos flojos superan a un informe corto con la respuesta exacta.
        flojos = [0.15] * 40
        preciso = [0.85]
        assert _puntaje_documento(flojos, "suma") > _puntaje_documento(preciso, "suma")
        assert _puntaje_documento(flojos, "max") < _puntaje_documento(preciso, "max")

    def test_topn_queda_entre_max_y_suma(self):
        puntajes = [0.9, 0.8, 0.1, 0.1, 0.1]
        top2 = _puntaje_documento(puntajes, "top2")
        assert _puntaje_documento(puntajes, "media") < top2
        assert top2 <= _puntaje_documento(puntajes, "max")

    def test_topn_no_revienta_con_menos_fragmentos_que_n(self):
        assert _puntaje_documento([0.5], "top3") == pytest.approx(0.5)

    def test_modo_desconocido_cae_en_max(self):
        assert _puntaje_documento([0.2, 0.9], "inventado") == pytest.approx(0.9)

    def test_la_agregacion_cambia_el_orden_de_los_documentos(self):
        cands = [candidato(f"c{i}", "LARGO", 0.15) for i in range(40)]
        cands.append(candidato("p1", "CORTO", 0.85))
        assert agregar_a_documentos(cands, top=1, modo="max") == ["CORTO"]
        assert agregar_a_documentos(cands, top=1, modo="suma") == ["LARGO"]


class TestFusionConvexa:
    def test_normaliza_cada_indice_por_separado(self):
        # mE5 comprime el coseno hacia 0,75-0,92 y BGE-M3 se mueve más abajo.
        # Sin normalizar, el índice de escala alta dominaría la suma; con ella,
        # el mejor de cada uno vale 1 y lo que pesa es la posición relativa.
        rankings = {
            "alto": [(meta("c1"), 0.92), (meta("c2"), 0.90)],
            "bajo": [(meta("c2"), 0.40), (meta("c1"), 0.10)],
        }
        puntajes = {c.chunk_id: c.puntaje for c in fusionar_convexa(rankings)}
        # c1 gana en 'alto' (1.0) y pierde en 'bajo' (0.0); c2 al revés.
        assert puntajes["c1"] == pytest.approx(1.0)
        assert puntajes["c2"] == pytest.approx(1.0)

    def test_conserva_la_magnitud_que_rrf_descarta(self):
        rankings = {
            "a": [(meta("c1"), 0.95), (meta("c2"), 0.20)],
            "b": [(meta("c1"), 0.93), (meta("c2"), 0.18)],
        }
        por_rrf = fusionar_rrf(rankings)
        por_convexa = fusionar_convexa(rankings)
        brecha_rrf = por_rrf[0].puntaje - por_rrf[1].puntaje
        brecha_cvx = por_convexa[0].puntaje - por_convexa[1].puntaje
        assert brecha_cvx > brecha_rrf

    def test_un_indice_con_todo_empatado_no_divide_por_cero(self):
        rankings = {"a": [(meta("c1"), 0.5), (meta("c2"), 0.5)]}
        puntajes = {c.chunk_id: c.puntaje for c in fusionar_convexa(rankings)}
        assert puntajes == {"c1": pytest.approx(1.0), "c2": pytest.approx(1.0)}

    def test_ranking_vacio_se_ignora(self):
        rankings = {"a": [(meta("c1"), 0.9)], "b": []}
        assert [c.chunk_id for c in fusionar_convexa(rankings)] == ["c1"]

    def test_propaga_el_idioma_como_las_otras_fusiones(self):
        rankings = {"a": [(meta("c1"), 0.9)]}
        assert fusionar_convexa(rankings)[0].idioma == "es"


class TestSignificanciaEnElInforme:
    """La tabla que decide qué se entrega tiene que traer el p-valor pareado.

    Comparar las diferencias de medias contra la anchura de un intervalo
    bootstrap es una regla de bolsillo que no distingue entre una ventaja
    pequeña y consistente —treinta y cuatro consultas de cincuenta— y una
    grande que viene de tres. El test pareado sí, y es la comparación que
    decide qué se entrega.
    """

    def corrida(self, mod, receta, por_consulta, ndcg=0.5, f1=0.5):
        return mod.Corrida(
            encoders=("bge-m3",), chunk=504, solape=0.15,
            cfg=dict(fusion="rrf", k0=60, boost=1.0, max_por_doc=3,
                     candidatos=200, umbral=None, subdividir=False,
                     agregacion="max"),
            ndcg=ndcg, f1=f1, ic_ndcg=(0.4, 0.6), ic_f1=(0.4, 0.6),
            receta=receta, por_consulta=por_consulta,
        )

    def test_una_ventaja_consistente_sale_con_p_pequeno(self, tmp_path):
        mod = cargar_barrido()
        qids = [f"q{i:03d}" for i in range(20)]
        base = self.corrida(
            mod, "entrega", {q: {"ndcg@10": 0.4, "f1@3": 0.4} for q in qids},
            ndcg=0.4, f1=0.4)
        mejor = self.corrida(
            mod, "candidata", {q: {"ndcg@10": 0.6, "f1@3": 0.6} for q in qids},
            ndcg=0.6, f1=0.6)

        mod.informar([base, mejor], top=10, carpeta=tmp_path)
        resumen = (tmp_path / "resumen.txt").read_text(encoding="utf-8")

        assert "p(NDCG" in resumen
        fila = [l for l in resumen.splitlines() if l.startswith("candidata")][-1]
        assert "0.0001" in fila or "0.000" in fila

    def test_una_ventaja_que_viene_de_una_sola_consulta_no_convence(self, tmp_path):
        mod = cargar_barrido()
        qids = [f"q{i:03d}" for i in range(20)]
        base = self.corrida(
            mod, "entrega", {q: {"ndcg@10": 0.5, "f1@3": 0.5} for q in qids})
        suerte = {q: {"ndcg@10": 0.5, "f1@3": 0.5} for q in qids}
        suerte["q000"] = {"ndcg@10": 1.0, "f1@3": 1.0}
        candidata = self.corrida(mod, "candidata", suerte, ndcg=0.525, f1=0.525)

        mod.informar([base, candidata], top=10, carpeta=tmp_path)
        resumen = (tmp_path / "resumen.txt").read_text(encoding="utf-8")

        fila = [l for l in resumen.splitlines() if l.startswith("candidata")][-1]
        p = float(fila.split()[-2])
        assert p > 0.05, fila

    def test_el_p_valor_llega_al_jsonl(self, tmp_path):
        mod = cargar_barrido()
        qids = [f"q{i:03d}" for i in range(10)]
        base = self.corrida(
            mod, "entrega", {q: {"ndcg@10": 0.3, "f1@3": 0.3} for q in qids},
            ndcg=0.3, f1=0.3)
        otra = self.corrida(
            mod, "candidata", {q: {"ndcg@10": 0.7, "f1@3": 0.7} for q in qids},
            ndcg=0.7, f1=0.7)

        mod.informar([base, otra], top=10, carpeta=tmp_path)
        filas = [json.loads(l) for l in
                 (tmp_path / "metricas.jsonl").read_text(encoding="utf-8").splitlines()]
        candidata = next(f for f in filas if f["receta"] == "candidata")
        assert candidata["p_ndcg@10"] < 0.05
        assert candidata["p_f1@3"] < 0.05

    def test_sin_receta_entrega_no_hay_con_que_comparar(self, tmp_path):
        """Una rejilla a mano no tiene línea base; el informe no debe inventarla."""
        mod = cargar_barrido()
        qids = [f"q{i:03d}" for i in range(10)]
        a = self.corrida(mod, None, {q: {"ndcg@10": 0.3, "f1@3": 0.3} for q in qids})
        b = self.corrida(mod, None, {q: {"ndcg@10": 0.7, "f1@3": 0.7} for q in qids})

        mod.informar([a, b], top=10, carpeta=tmp_path)
        resumen = (tmp_path / "resumen.txt").read_text(encoding="utf-8")
        assert "p(NDCG" not in resumen


class TestFusionCombMNZ:
    """CombMNZ es una de las tres fusiones que la §8.4 nombra y era la que
    faltaba. Frente a CombSUM añade un factor: el número de índices en los que
    el fragmento aparece, que premia el consenso entre encoders sobre la
    puntuación alta en uno solo.
    """

    def test_un_solo_indice_no_cambia_el_orden(self):
        from aphelion.busqueda.recuperacion import fusionar_combmnz

        rankings = {"a": [(meta("c1"), 0.9), (meta("c2"), 0.4)]}
        assert [c.chunk_id for c in fusionar_combmnz(rankings)] == ["c1", "c2"]

    def test_el_consenso_multiplica_la_suma(self):
        from aphelion.busqueda.recuperacion import fusionar_combmnz

        # c1 aparece en los dos índices, c2 solo en uno con mejor puntuación.
        rankings = {
            "a": [(meta("c2"), 0.90), (meta("c1"), 0.30)],
            "b": [(meta("c1"), 0.25)],
        }
        puntajes = {c.chunk_id: c.puntaje for c in fusionar_combmnz(rankings)}
        assert puntajes["c1"] == pytest.approx((0.30 + 0.25) * 2)
        assert puntajes["c2"] == pytest.approx(0.90 * 1)

    def test_el_consenso_puede_adelantar_a_una_puntuacion_mas_alta(self):
        """Es la diferencia con CombSUM y la razón de que exista."""
        from aphelion.busqueda.recuperacion import fusionar_combmnz, fusionar_combsum

        rankings = {
            "a": [(meta("c2"), 0.90), (meta("c1"), 0.30)],
            "b": [(meta("c1"), 0.25)],
        }
        assert [c.chunk_id for c in fusionar_combsum(rankings)] == ["c2", "c1"]
        assert [c.chunk_id for c in fusionar_combmnz(rankings)] == ["c1", "c2"]

    def test_registra_las_posiciones_como_las_otras_fusiones(self):
        from aphelion.busqueda.recuperacion import fusionar_combmnz

        rankings = {"a": [(meta("c1"), 0.9)], "b": [(meta("c1"), 0.8)]}
        assert fusionar_combmnz(rankings)[0].posiciones == {"a": 1, "b": 1}

    def test_propaga_el_idioma(self):
        from aphelion.busqueda.recuperacion import fusionar_combmnz

        assert fusionar_combmnz({"a": [(meta("c1"), 0.9)]})[0].idioma == "es"

    def test_ranking_vacio_se_ignora(self):
        from aphelion.busqueda.recuperacion import fusionar_combmnz

        rankings = {"a": [(meta("c1"), 0.9)], "b": []}
        assert [c.chunk_id for c in fusionar_combmnz(rankings)] == ["c1"]

    def test_ordenar_acepta_combmnz_como_fusion(self):
        from aphelion.busqueda.recuperacion import fusionar_combmnz, fusionar_combsum

        rankings = {
            "a": [(meta("c2", "D2"), 0.90), (meta("c1", "D1"), 0.30)],
            "b": [(meta("c1", "D1"), 0.25)],
        }
        # Si `ordenar` no conociera 'combmnz' caería en RRF por el else final y
        # el resultado no se distinguiría del de otra fusión cualquiera.
        esperado = [c.chunk_id for c in fusionar_combmnz(rankings)]
        distinto = [c.chunk_id for c in fusionar_combsum(rankings)]
        assert esperado != distinto  # el caso de prueba discrimina

        rec = _recuperador_de_prueba()
        obtenido = rec.ordenar(rankings, _consulta(), fusion="combmnz")
        assert [c.chunk_id for c in obtenido.fragmentos] == esperado


def _consulta():
    from aphelion.busqueda.consultas import Consulta

    return Consulta(query_id="q001", texto="da igual")


def _recuperador_de_prueba():
    from aphelion.busqueda.recuperacion import Recuperador
    from aphelion.indice import vectores as mod_vec
    import numpy as np

    indice = mod_vec.IndiceVectorial(
        "a", mod_vec.construir(np.eye(2, dtype=np.float32), 2),
        [meta("c1"), meta("c2")])
    return Recuperador({"a": indice})
