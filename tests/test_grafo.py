"""Grafo de conocimiento: canonicalización, tripletas, esquema y GraphML.

La prueba que justifica el archivo entero es `test_exportar_ida_y_vuelta`: el
escritor GraphML de NetworkX solo admite `int`, `str`, `float` y `bool`, y falla
**al exportar**, no al construir. Sin esta prueba, un atributo de tipo lista
sobrevive a toda la construcción del grafo sobre 64.484 fragmentos y revienta en
el último segundo de la etapa más cara del bonus.
"""

from __future__ import annotations

import pytest

from aphelion.grafo import busqueda, construccion, entidades as ent, relaciones as rel

nx = pytest.importorskip("networkx")


def mencion(chunk_id, doc_id, texto, tipo="organizacion", inicio=0, fin=None):
    return ent.Mencion(
        chunk_id=chunk_id,
        doc_id=doc_id,
        texto=texto,
        tipo=tipo,
        inicio=inicio,
        fin=fin if fin is not None else inicio + len(texto),
        confianza=0.9,
    )


def fragmento(chunk_id, doc_id, texto, fenomeno=1, posicion=0):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "texto": texto,
        "fuente": f"{doc_id}.pdf",
        "formato": "pdf",
        "fenomeno": fenomeno,
        "posicion": posicion,
        "idioma": "es",
    }


# --- Normalización y canonicalización -------------------------------------


@pytest.mark.parametrize(
    "crudo, esperado",
    [
        ("la Unión Europea", "union europea"),
        ("The European Union", "european union"),
        ("EE. UU.", "ee uu"),
        ("  Fuerza  Aeroespacial\nColombiana ", "fuerza aeroespacial colombiana"),
        ("a NASA", "nasa"),  # artículo portugués
    ],
)
def test_normalizar_quita_articulo_tildes_y_puntuacion(crudo, esperado):
    assert ent.normalizar_nombre(crudo) == esperado


def test_articulo_solo_cae_si_hay_algo_detras():
    # «La» sola es un nombre de una palabra, no un artículo suelto: quitarlo
    # dejaría la clave vacía y la mención desaparecería del grafo sin aviso.
    assert ent.normalizar_nombre("La") == "la"


def test_canonicalizar_agrupa_y_elige_la_forma_mas_frecuente():
    menciones = [
        mencion("c1", "d1", "la NASA"),
        mencion("c2", "d2", "NASA"),
        mencion("c3", "d3", "NASA"),
        mencion("c4", "d1", "Agencia Espacial Europea"),
    ]
    entidades, clave_a_id = ent.canonicalizar(menciones)

    assert set(entidades) == {"nasa", "agencia espacial europea"}
    assert entidades["nasa"].nombre == "NASA"  # 2 apariciones contra 1 de «la NASA»
    assert entidades["nasa"].frecuencia_documental == 3
    assert clave_a_id["nasa"] == "nasa"


def test_canonicalizar_es_determinista_ante_empate():
    # Dos formas igual de frecuentes: sin desempate alfabético, el nombre elegido
    # dependería del orden del diccionario y el grafo dejaría de reproducirse.
    menciones = [mencion("c1", "d1", "ONU"), mencion("c2", "d2", "onu")]
    primero = ent.canonicalizar(menciones)[0]["onu"].nombre
    for _ in range(5):
        assert ent.canonicalizar(menciones)[0]["onu"].nombre == primero


def test_canonicalizar_aplica_equivalencias_entre_idiomas():
    menciones = [
        mencion("c1", "d1", "Estados Unidos"),
        mencion("c2", "d2", "United States"),
    ]
    entidades, clave_a_id = ent.canonicalizar(
        menciones, {"united states": "estados unidos"}
    )

    assert list(entidades) == ["estados unidos"]
    assert clave_a_id["united states"] == "estados unidos"
    assert entidades["estados unidos"].frecuencia_documental == 2
    assert "united states" in entidades["estados unidos"].alias


def test_agrupar_por_embedding_une_solo_lo_que_pasa_el_umbral():
    import numpy as np

    # Encoder falso: los dos primeros nombres apuntan casi a la misma dirección,
    # el tercero es ortogonal.
    vectores = {
        "estados unidos": [1.0, 0.0],
        "united states": [0.9997, 0.0245],
        "brasil": [0.0, 1.0],
    }

    def codificar(textos):
        return np.asarray([vectores[t] for t in textos], dtype=np.float32)

    equivalencias = ent.agrupar_por_embedding(
        ["brasil", "estados unidos", "united states"], codificar
    )
    assert equivalencias == {"united states": "estados unidos"}


def test_agrupar_por_embedding_no_construye_la_matriz_completa():
    """Con muchos nombres, la versión cuadrática pedía gigabytes.

    2.000 nombres son 4 millones de similitudes, que aún caben; los ~70.000 del
    corpus completo son 19 GB y no. Esta prueba no mide memoria: comprueba que la
    función sigue en pie con un tamaño donde la matriz completa ya sería mala
    idea, que es lo que rompería si alguien volviera a `vectores @ vectores.T`.
    """
    import numpy as np

    generador = np.random.default_rng(20260801)
    vectores = generador.normal(size=(2000, 16)).astype(np.float32)
    vectores /= np.linalg.norm(vectores, axis=1, keepdims=True)
    # Los dos últimos son casi el mismo punto: tiene que salir ese par y solo ese.
    vectores[-1] = vectores[-2]

    claves = [f"entidad {i:05d}" for i in range(len(vectores))]
    equivalencias = ent.agrupar_por_embedding(claves, lambda _: vectores)

    assert equivalencias == {claves[-1]: claves[-2]}


def test_admisible_descarta_ruido_tabular():
    assert ent.admisible("Fuerza Aérea")
    assert not ent.admisible("X")  # una letra
    assert not ent.admisible("2024")  # cifra de tabla
    assert not ent.admisible(" ".join(["palabra"] * 9))  # media frase


# --- Caché ----------------------------------------------------------------


def test_la_cache_separa_los_backends():
    from aphelion import config

    # Comprobar el esquema con `--ner falso` no puede dejar una caché que la
    # corrida con el modelo real reutilice: el grafo del entregable saldría de un
    # reconocedor de mayúsculas y nada avisaría.
    assert config.ruta_entidades("falso") != config.ruta_entidades("onnx")


def test_cache_de_entidades_ida_y_vuelta(tmp_path):
    original = {"c1": [mencion("c1", "d1", "NASA", inicio=10)]}
    ruta = ent.escribir_cache(original, tmp_path / "entidades.jsonl")
    leido = ent.leer_cache(ruta)

    assert leido["c1"][0].texto == "NASA"
    assert leido["c1"][0].inicio == 10
    assert leido["c1"][0].doc_id == "d1"


# --- Relaciones -----------------------------------------------------------


def _tripletas(texto, pares):
    menciones = [
        mencion("c1", "d1", t, inicio=texto.index(t), fin=texto.index(t) + len(t))
        for t in pares
    ]
    return rel.extraer(texto, menciones, [texto])


def test_relacion_activa_conserva_la_direccion():
    texto = "Estados Unidos desarrolla sistemas autónomos de armas."
    tripletas = _tripletas(texto, ["Estados Unidos", "sistemas autónomos"])

    assert len(tripletas) == 1
    assert tripletas[0].clave == ("estados unidos", "desarrolla", "sistemas autonomos")


def test_relacion_pasiva_invierte_sujeto_y_objeto():
    texto = "El sistema autónomo está regulado por la Convención de Ginebra."
    tripletas = _tripletas(texto, ["sistema autónomo", "Convención de Ginebra"])

    assert len(tripletas) == 1
    # (Y, regula, X), no (X, regula, Y): la pasiva se comprueba antes que la
    # activa justo para esto.
    assert tripletas[0].sujeto == "convencion de ginebra"
    assert tripletas[0].tipo == "regula"
    assert tripletas[0].objeto == "sistema autonomo"


def test_la_tripleta_conserva_su_procedencia():
    texto = "La ESA opera el satélite Sentinel."
    tripletas = _tripletas(texto, ["ESA", "satélite Sentinel"])

    assert tripletas[0].doc_id == "d1"
    assert tripletas[0].chunk_id == "c1"
    assert "opera" in tripletas[0].evidencia


def test_sin_patron_no_hay_tripleta():
    # Coocurrir en una oración no es una relación: sin verbo del inventario en
    # medio, el par no produce arista.
    texto = "Colombia y Brasil, según el informe anual del observatorio."
    assert _tripletas(texto, ["Colombia", "Brasil"]) == []


def test_menciones_lejanas_no_se_relacionan():
    relleno = "y por consiguiente, tal como se ha venido señalando en el capítulo,"
    texto = f"Colombia {relleno} desarrolla programas espaciales."
    assert _tripletas(texto, ["Colombia", "programas espaciales"]) == []


def test_sustantivo_no_se_confunde_con_verbo():
    # `desarroll\w+` habría emparejado «desarrollo» y producido una relación que
    # el texto no afirma. Por eso las conjugaciones están enumeradas.
    texto = "El informe analiza el desarrollo de Colombia en el sector espacial."
    assert _tripletas(texto, ["Colombia", "sector espacial"]) == []


def test_entidad_no_se_relaciona_consigo_misma():
    texto = "La NASA financia a la NASA en el programa."
    menciones = [
        mencion("c1", "d1", "NASA", inicio=3, fin=7),
        mencion("c1", "d1", "NASA", inicio=25, fin=29),
    ]
    assert rel.extraer(texto, menciones, [texto]) == []


def test_las_tripletas_no_cruzan_frontera_de_oracion():
    texto = "Colombia lo confirmó. La ONU regula el uso militar."
    menciones = [
        mencion("c1", "d1", "Colombia", inicio=0, fin=8),
        mencion("c1", "d1", "ONU", inicio=25, fin=28),
        mencion("c1", "d1", "uso militar", inicio=37, fin=48),
    ]
    tripletas = rel.extraer(
        texto, menciones, ["Colombia lo confirmó.", "La ONU regula el uso militar."]
    )
    assert [t.clave for t in tripletas] == [("onu", "regula", "uso militar")]


# --- Construcción y poda --------------------------------------------------


def _corpus_minimo():
    """Dos documentos, tres fragmentos, tres entidades con frecuencias distintas."""
    fragmentos = [
        fragmento("c1", "d1", "Estados Unidos desarrolla sistemas autónomos."),
        fragmento("c2", "d2", "La ONU regula los sistemas autónomos.", posicion=1),
        fragmento("c3", "d2", "Un apunte suelto sobre Zzz.", posicion=2),
    ]
    menciones = {
        "c1": [
            mencion("c1", "d1", "Estados Unidos", inicio=0, fin=14),
            mencion("c1", "d1", "sistemas autónomos", inicio=26, fin=44),
        ],
        "c2": [
            mencion("c2", "d2", "ONU", inicio=3, fin=6),
            mencion("c2", "d2", "sistemas autónomos", inicio=18, fin=36),
        ],
        "c3": [mencion("c3", "d2", "Zzz", inicio=23, fin=26)],
    }
    todas = [m for lista in menciones.values() for m in lista]
    entidades, clave_a_id = ent.canonicalizar(todas)
    return fragmentos, menciones, entidades, clave_a_id


def test_poda_descarta_la_entidad_de_un_solo_documento():
    _, _, entidades, _ = _corpus_minimo()
    # `zzz` aparece en un documento; `sistemas autonomos` en dos.
    admitidas = construccion.entidades_admitidas(entidades, n_documentos=2, max_fraccion=1.0)

    assert "sistemas autonomos" in admitidas
    assert "zzz" not in admitidas


def test_poda_descarta_la_entidad_omnipresente():
    _, _, entidades, _ = _corpus_minimo()
    entidades["ubicua"] = ent.Entidad("ubicua", "Ubicua", "organizacion", ("ubicua",), 90)
    admitidas = construccion.entidades_admitidas(entidades, n_documentos=100)

    assert "sistemas autonomos" in admitidas  # df=2, dentro de la ventana
    assert "ubicua" not in admitidas  # df=90 > 5% de 100


def test_la_ventana_de_poda_nunca_queda_vacia():
    # Con pocos documentos, el 5% redondea a 0 y `2 <= df <= 0` no lo cumple
    # nadie: el grafo saldría vacío sin que nada fallara. El tope se ancla al
    # mínimo justo para que eso no pueda pasar en silencio.
    _, _, entidades, _ = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, n_documentos=2)

    assert "sistemas autonomos" in admitidas


def test_construir_arma_el_esquema_completo():
    fragmentos, menciones, entidades, clave_a_id = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, 2, min_df=1, max_fraccion=1.0)
    tripletas = [
        rel.Tripleta("estados unidos", "desarrolla", "sistemas autonomos", "d1", "c1", "…")
    ]
    grafo = construccion.construir(fragmentos, menciones, admitidas, clave_a_id, tripletas)

    resumen = construccion.resumen(grafo)
    assert resumen["documentos"] == 2
    assert resumen["fragmentos"] == 3
    assert resumen["relaciones"] == 1
    assert grafo.has_edge("C:c1", "E:estados unidos")
    assert grafo["C:c1"]["E:estados unidos"]["relacion"] == "MENCIONA"
    assert grafo.has_edge("D:d1", "C:c1")


def test_el_fragmento_sin_entidades_admitidas_no_entra():
    fragmentos, menciones, entidades, clave_a_id = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, 2, max_fraccion=1.0)
    grafo = construccion.construir(fragmentos, menciones, admitidas, clave_a_id, [])

    # c3 solo mencionaba a `zzz`, que la poda descartó: un nodo de fragmento
    # aislado pesa igual en el XML y no se alcanza por ningún camino.
    assert "C:c3" not in grafo
    assert "C:c1" in grafo


def test_la_tripleta_huerfana_no_inventa_nodos():
    fragmentos, menciones, entidades, clave_a_id = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, 2, max_fraccion=1.0)
    huerfana = [rel.Tripleta("no existe", "regula", "sistemas autonomos", "d1", "c1", "…")]
    grafo = construccion.construir(fragmentos, menciones, admitidas, clave_a_id, huerfana)

    assert "E:no existe" not in grafo
    assert construccion.resumen(grafo)["relaciones"] == 0


def test_la_tripleta_repetida_cuenta_apariciones():
    fragmentos, menciones, entidades, clave_a_id = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, 2, min_df=1, max_fraccion=1.0)
    dos_veces = [
        rel.Tripleta("estados unidos", "desarrolla", "sistemas autonomos", "d1", "c1", "a"),
        rel.Tripleta("estados unidos", "desarrolla", "sistemas autonomos", "d2", "c2", "b"),
    ]
    grafo = construccion.construir(fragmentos, menciones, admitidas, clave_a_id, dos_veces)

    arista = grafo["E:estados unidos"]["E:sistemas autonomos"]
    assert arista["apariciones"] == 2
    assert arista["evidencia"] == "a"  # se conserva la primera


def test_tope_de_entidades_por_fragmento():
    frag = [fragmento("c1", "d1", "texto")]
    menciones = {
        "c1": [mencion("c1", "d1", f"Entidad {i}", inicio=i) for i in range(20)]
    }
    todas = menciones["c1"]
    entidades, clave_a_id = ent.canonicalizar(todas)
    grafo = construccion.construir(
        frag, menciones, entidades, clave_a_id, [], max_por_fragmento=5
    )
    assert grafo.out_degree("C:c1") == 5


# --- GraphML --------------------------------------------------------------


def test_validar_escalares_localiza_el_atributo_culpable():
    grafo = nx.DiGraph()
    grafo.add_node("E:x", tipo_nodo="entidad", chunks=["c1", "c2"])

    problemas = construccion.validar_escalares(grafo)
    assert len(problemas) == 1
    assert "E:x" in problemas[0] and "chunks" in problemas[0] and "list" in problemas[0]


def test_exportar_rechaza_lo_que_graphml_no_sabe_escribir(tmp_path):
    grafo = nx.DiGraph()
    grafo.add_node("E:x", tipo_nodo="entidad", alias=["a", "b"])

    destino = tmp_path / "grafo.graphml"
    with pytest.raises(TypeError, match="no serializables"):
        construccion.exportar(grafo, destino)
    assert not destino.exists()  # no deja un archivo a medias


def test_exportar_ida_y_vuelta(tmp_path):
    """El grafo real se escribe y se relee sin perder nada.

    Es la prueba que cierra el riesgo: si algún atributo del esquema deja de ser
    escalar —los alias de una entidad son varios y la tentación de guardarlos como
    lista es permanente—, esto falla en milisegundos en vez de al final de la
    etapa más cara.
    """
    fragmentos, menciones, entidades, clave_a_id = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, 2, min_df=1, max_fraccion=1.0)
    tripletas = [
        rel.Tripleta(
            "estados unidos", "desarrolla", "sistemas autonomos", "d1", "c1",
            "Estados Unidos desarrolla sistemas autónomos.",
        )
    ]
    grafo = construccion.construir(fragmentos, menciones, admitidas, clave_a_id, tripletas)

    destino = construccion.exportar(grafo, tmp_path / "grafo" / "grafo.graphml")
    assert destino.exists()

    releido = nx.read_graphml(destino)
    assert releido.number_of_nodes() == grafo.number_of_nodes()
    assert releido.number_of_edges() == grafo.number_of_edges()

    # La procedencia que exige la §7.2 sobrevive al viaje.
    arista = releido["E:estados unidos"]["E:sistemas autonomos"]
    assert arista["doc_id"] == "d1"
    assert arista["chunk_id"] == "c1"
    assert arista["tipo"] == "desarrolla"

    # Y los tipos numéricos vuelven como números, no como cadenas.
    assert releido.nodes["C:c1"]["fenomeno"] == 1


def test_alias_viajan_como_cadena_recuperable(tmp_path):
    menciones = [
        mencion("c1", "d1", "Estados Unidos"),
        mencion("c2", "d2", "United States"),
    ]
    entidades, clave_a_id = ent.canonicalizar(
        menciones, {"united states": "estados unidos"}
    )
    grafo = construccion.construir(
        [fragmento("c1", "d1", "t"), fragmento("c2", "d2", "t")],
        {"c1": [menciones[0]], "c2": [menciones[1]]},
        entidades,
        clave_a_id,
        [],
    )
    releido = nx.read_graphml(construccion.exportar(grafo, tmp_path / "g.graphml"))

    alias = busqueda.indice_de_alias(releido)
    assert alias["united states"] == "E:estados unidos"
    assert alias["estados unidos"] == "E:estados unidos"


# --- Canal de recuperación ------------------------------------------------


def _grafo_de_busqueda():
    fragmentos, menciones, entidades, clave_a_id = _corpus_minimo()
    admitidas = construccion.entidades_admitidas(entidades, 2, min_df=1, max_fraccion=1.0)
    tripletas = [
        rel.Tripleta("estados unidos", "desarrolla", "sistemas autonomos", "d1", "c1", "…")
    ]
    return construccion.construir(fragmentos, menciones, admitidas, clave_a_id, tripletas)


def test_entidades_de_consulta_resuelve_por_forma_normalizada():
    grafo = _grafo_de_busqueda()
    alias = busqueda.indice_de_alias(grafo)

    assert busqueda.entidades_de_consulta(["los Estados Unidos"], alias) == [
        "E:estados unidos"
    ]
    assert busqueda.entidades_de_consulta(["Marte"], alias) == []


def test_ranking_devuelve_los_fragmentos_que_mencionan_la_entidad():
    grafo = _grafo_de_busqueda()
    ranking = busqueda.ranking_por_entidades(grafo, ["E:onu"], peso_vecino=0.0)

    assert [chunk for chunk, _ in ranking] == ["c2"]


def test_el_vecino_de_primer_orden_puntua_menos_que_la_mencion_directa():
    grafo = _grafo_de_busqueda()
    # `estados unidos` está relacionada con `sistemas autonomos`, que menciona c2.
    ranking = dict(busqueda.ranking_por_entidades(grafo, ["E:estados unidos"], peso_vecino=0.5))

    assert ranking["c1"] > ranking["c2"]  # c1 la menciona; c2 solo al vecino
    assert set(ranking) == {"c1", "c2"}


def test_sin_salto_a_vecinos_solo_queda_la_mencion_directa():
    grafo = _grafo_de_busqueda()
    ranking = busqueda.ranking_por_entidades(grafo, ["E:estados unidos"], peso_vecino=0.0)
    assert [chunk for chunk, _ in ranking] == ["c1"]


def test_consulta_sin_entidades_no_devuelve_candidatos():
    assert busqueda.ranking_por_entidades(_grafo_de_busqueda(), []) == []


def test_ranking_para_fusion_descarta_lo_que_el_indice_no_tiene():
    grafo = _grafo_de_busqueda()
    # Solo c1 tiene metadata: c2 vendría del grafo pero no del índice.
    meta = {"c1": {"chunk_id": "c1", "doc_id": "d1"}}
    salida = busqueda.ranking_para_fusion(grafo, ["E:estados unidos"], meta, peso_vecino=0.5)

    assert [m["chunk_id"] for m, _ in salida] == ["c1"]


def test_ranking_para_fusion_encaja_con_fusionar_rrf():
    """El canal del grafo entra en la fusión existente sin tocarla."""
    from aphelion.busqueda import recuperacion

    grafo = _grafo_de_busqueda()
    meta = {
        "c1": {"chunk_id": "c1", "doc_id": "d1", "texto": "t", "fuente": "d1.pdf", "fenomeno": 1},
        "c2": {"chunk_id": "c2", "doc_id": "d2", "texto": "t", "fuente": "d2.pdf", "fenomeno": 1},
    }
    rankings = {
        "bge-m3": [(meta["c2"], 0.8)],
        "grafo": busqueda.ranking_para_fusion(grafo, ["E:estados unidos"], meta, peso_vecino=0.5),
    }
    candidatos = recuperacion.fusionar_rrf(rankings)

    # c2 lo traen los dos canales; c1 solo el grafo. El consenso manda.
    assert candidatos[0].chunk_id == "c2"
    assert candidatos[0].consenso == 2
    assert {c.chunk_id for c in candidatos} == {"c1", "c2"}
