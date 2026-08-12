"""El canal de recuperación del grafo (§8.5), para medirlo en el barrido.

La §8.5 describe cuatro pasos: NER sobre la consulta, recuperar los chunks
vinculados a esas entidades y a sus vecinos de primer orden, puntuarlos por el
número de relaciones relevantes, y fusionarlos con los resultados vectoriales
«tratando el grafo como un índice adicional».

Ese último punto es el que hace que esto encaje sin tocar nada:
`recuperacion.fusionar_rrf` recibe `dict[str, list[tuple[dict, float]]]` —un
canal y su lista ordenada—, así que el grafo entra como una entrada más del
diccionario y la fusión no se entera.

    rankings = recuperador.buscar(consulta)
    rankings["grafo"] = ranking_para_fusion(grafo, entidades, indice_meta, k)
    candidatos = recuperacion.fusionar_rrf(rankings, k0)

**RRF y no las otras fusiones.** El puntaje de este canal cuenta relaciones, no
cosenos. RRF opera sobre posiciones y es inmune a la escala —por eso lo propone
la propia §8.5—, mientras que `fusionar_convexa` o `fusionar_combsum` sumarían
magnitudes que no significan lo mismo. El entregable fusiona con RRF, así que
esto sale gratis; si algún día cambia, este canal necesita normalización antes.

**Dónde iría dentro del `Recuperador`.** En `buscar` y no en `ordenar`: la
consulta al grafo depende de la consulta y no de los hiperparámetros de fusión,
así que pertenece a la mitad cara, la que el barrido cachea una vez para
reordenar cien veces. Nada de esto está conectado todavía —ver el docstring del
paquete—, pero la firma está pensada para que conectarlo sea añadir una línea.
"""

from __future__ import annotations

from .. import config
from .entidades import normalizar_nombre


def indice_de_alias(grafo) -> dict[str, str]:
    """clave normalizada -> id de nodo de entidad, incluidos los alias.

    Los alias viajan en el GraphML como cadena separada por barras verticales
    porque GraphML no admite listas; esto los devuelve a un diccionario. Se
    construye una vez por corrida, no por consulta.
    """
    indice: dict[str, str] = {}
    for nodo, datos in grafo.nodes(data=True):
        if datos.get("tipo_nodo") != "entidad":
            continue
        indice[nodo[2:]] = nodo  # la clave canónica, sin el prefijo "E:"
        for alias in (datos.get("alias") or "").split("|"):
            if alias:
                indice.setdefault(alias, nodo)
    return indice


def entidades_de_consulta(menciones_texto: list[str], alias: dict[str, str]) -> list[str]:
    """Los nodos del grafo que nombra la consulta.

    Recibe las formas superficiales que detectó el NER —el paso 1 de la §8.5— y
    no el texto crudo, para que el barrido pueda medir el canal con entidades
    anotadas a mano y separar dos preguntas que si no se mezclan: si el grafo
    ayuda, y si el NER acierta sobre consultas de una línea.
    """
    vistos: dict[str, None] = {}
    for texto in menciones_texto:
        nodo = alias.get(normalizar_nombre(texto))
        if nodo:
            vistos.setdefault(nodo, None)
    return list(vistos)


def ranking_por_entidades(
    grafo,
    entidades: list[str],
    k: int = config.GRAFO_CANDIDATOS,
    peso_vecino: float = config.GRAFO_PESO_VECINO,
) -> list[tuple[str, float]]:
    """chunk_id -> puntaje, ordenado, para las entidades que nombra la consulta.

    Un fragmento puntúa por cada entidad de la consulta que menciona, y menos —en
    proporción a `peso_vecino`— por cada vecino de primer orden de esas entidades.
    Los vecinos entran porque es lo que distingue a un grafo de una lista
    invertida: la consulta que pregunta por basura orbital debería alcanzar los
    fragmentos que hablan de las misiones que la generan aunque no repitan el
    término.

    El descuento por vecino no está medido sobre este corpus. Es la primera
    dimensión que conviene abrir en el barrido, junto con `k`.
    """
    if not entidades:
        return []

    directas = {e: 1.0 for e in entidades}
    alcance = dict(directas)

    if peso_vecino > 0:
        for entidad in entidades:
            # Vecinos por relaciones tipadas, en los dos sentidos: «X regula Y» y
            # «Y regulado por X» describen la misma vecindad y la dirección de la
            # arista no debería decidir si se alcanza.
            vecinos = set(grafo.successors(entidad)) | set(grafo.predecessors(entidad))
            for vecino in vecinos:
                if grafo.nodes[vecino].get("tipo_nodo") != "entidad":
                    continue
                if vecino in directas:
                    continue
                alcance[vecino] = max(alcance.get(vecino, 0.0), peso_vecino)

    puntajes: dict[str, float] = {}
    for entidad, peso in alcance.items():
        for fragmento in grafo.predecessors(entidad):
            datos_arista = grafo[fragmento][entidad]
            if datos_arista.get("relacion") != "MENCIONA":
                continue
            # El número de menciones cuenta, pero con rendimientos decrecientes:
            # un fragmento que repite la entidad diez veces no es diez veces más
            # relevante, y sin amortiguar acapararía el canal entero.
            n = int(datos_arista.get("n_menciones", 1))
            puntajes[fragmento] = puntajes.get(fragmento, 0.0) + peso * (1.0 + (n - 1) * 0.25)

    # Desempate alfabético para que el canal sea reproducible: dos fragmentos con
    # el mismo puntaje deben salir siempre en el mismo orden.
    ordenados = sorted(puntajes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(nodo[2:], puntaje) for nodo, puntaje in ordenados[:k]]


def ranking_para_fusion(
    grafo,
    entidades: list[str],
    meta_por_chunk: dict[str, dict],
    k: int = config.GRAFO_CANDIDATOS,
    peso_vecino: float = config.GRAFO_PESO_VECINO,
) -> list[tuple[dict, float]]:
    """El mismo ranking, con la forma que espera `fusionar_rrf`.

    Los fragmentos que el grafo conoce pero el índice no —no debería haberlos, y
    los hay si el grafo se construyó con otra fragmentación— se descartan en
    silencio: fusionar un candidato sin metadata haría estallar la construcción de
    la salida mucho más tarde y con un error que no señalaría aquí.
    """
    salida: list[tuple[dict, float]] = []
    for chunk_id, puntaje in ranking_por_entidades(grafo, entidades, k, peso_vecino):
        meta = meta_por_chunk.get(chunk_id)
        if meta is not None:
            salida.append((meta, puntaje))
    return salida
