"""Etapa 3 del grafo: construcción con NetworkX y exportación a GraphML.

**El esquema.** Tres tipos de nodo y tres de arista bastan para cubrir a la vez
la §7.2 (tripletas con procedencia) y la §7.3 (entidad ↔ chunks que la mencionan):

    documento --CONTIENE--> fragmento --MENCIONA--> entidad
                                       entidad --RELACION--> entidad

Los identificadores llevan prefijo (`E:`, `C:`, `D:`) porque GraphML tiene un
único espacio de nombres para los nodos: sin prefijo, una entidad llamada igual
que un `doc_id` los fundiría en uno.

**Por qué la procedencia va en la arista y no en el nodo.** Además de ser lo que
pide la §7.2, resuelve el problema de serialización: el escritor GraphML de
NetworkX solo admite `int`, `str`, `float` y `bool`, y guardar «los chunks donde
aparece esta entidad» como lista dentro del nodo levanta
`TypeError: GraphML does not support type <class 'list'> as data values`. Como
arista, cada mención es su propia fila y no hay nada que serializar a mano.
`exportar` lo comprueba antes de escribir para que el fallo diga qué atributo lo
causó y no solo qué tipo.

**La poda es parte del diseño, no una optimización.** 64.484 fragmentos por unas
pocas entidades cada uno son cientos de miles de aristas, y GraphML es XML: sin
podar, el archivo se va a cientos de megas y `nx.read_graphml` tarda minutos. Lo
que se descarta es además lo que no aporta caminos: una entidad que aparece en un
solo documento no conecta nada, y una que aparece en la mitad del corpus es
membrete institucional, no señal.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from .entidades import Entidad, Mencion
from .relaciones import Tripleta

# Tipos admitidos por el escritor GraphML de NetworkX. `None` no está: un atributo
# nulo también rompe la exportación, así que se normaliza a cadena vacía.
ESCALARES = (str, int, float, bool)


def _limpiar(atributos: dict) -> dict:
    """Deja los atributos en tipos que GraphML acepta.

    Solo toca `None` -> `""`. Cualquier otro tipo no admitido se deja pasar para
    que `validar_escalares` lo denuncie con nombre y apellidos: convertir a
    cadena en silencio escondería un error de esquema hasta que alguien leyera el
    grafo y encontrara `"['a', 'b']"` donde esperaba una lista.
    """
    return {k: ("" if v is None else v) for k, v in atributos.items()}


def validar_escalares(grafo) -> list[str]:
    """Los atributos que GraphML no sabe escribir, con su ubicación.

    Se llama antes de escribir porque `write_graphml` falla a mitad de archivo y
    su mensaje dice el tipo pero no dónde estaba. Con 64.484 fragmentos, esa
    diferencia son diez minutos de búsqueda o ninguno.
    """
    problemas: list[str] = []
    for nodo, datos in grafo.nodes(data=True):
        for clave, valor in datos.items():
            if not isinstance(valor, ESCALARES):
                problemas.append(f"nodo {nodo}: {clave} es {type(valor).__name__}")
    for origen, destino, datos in grafo.edges(data=True):
        for clave, valor in datos.items():
            if not isinstance(valor, ESCALARES):
                problemas.append(
                    f"arista {origen} -> {destino}: {clave} es {type(valor).__name__}"
                )
    return problemas


def entidades_admitidas(
    entidades: dict[str, Entidad],
    n_documentos: int,
    min_df: int = config.GRAFO_MIN_DF,
    max_fraccion: float = config.GRAFO_MAX_FRACCION_DF,
) -> dict[str, Entidad]:
    """Descarta las entidades que no pueden aportar caminos útiles.

    Por abajo, las que aparecen en menos de `min_df` documentos: una entidad
    citada una sola vez no conecta dos fragmentos, que es lo único que el canal
    del grafo sabe hacer. Por arriba, las que aparecen en más de `max_fraccion`
    del corpus: son «Colombia» en un corpus colombiano o «AI» en el fenómeno 1,
    y traen de vuelta medio índice sin discriminar nada.

    **El tope nunca baja del mínimo.** Sobre el corpus entero (1.826 documentos)
    el 5% son 91 y las dos cotas no se estorban, pero sobre una muestra de diez
    documentos el tope sale 0 y la ventana `2 <= df <= 0` no la cumple nadie: el
    grafo se queda vacío sin que nada falle. Pasó en la primera corrida de prueba
    con `--limite`. Ese silencio es peor que el error, así que el tope se
    ancla al mínimo: en un corpus pequeño la cota superior simplemente no ata.
    """
    tope = max(min_df, int(n_documentos * max_fraccion))
    return {
        clave: entidad
        for clave, entidad in entidades.items()
        if min_df <= entidad.frecuencia_documental <= tope
    }


def construir(
    fragmentos: list[dict],
    menciones_por_chunk: dict[str, list[Mencion]],
    entidades: dict[str, Entidad],
    clave_a_id: dict[str, str],
    tripletas: list[Tripleta],
    max_por_fragmento: int = config.GRAFO_MAX_ENTIDADES_POR_FRAGMENTO,
):
    """El grafo dirigido completo, ya podado.

    `fragmentos` son los registros del índice (mismos campos que
    `metadata.jsonl`). Solo entran al grafo los que conservan alguna mención tras
    la poda: un nodo de fragmento aislado pesa lo mismo en el XML que uno
    conectado y no se puede alcanzar por ningún camino.
    """
    import networkx as nx

    grafo = nx.DiGraph()
    grafo.graph["descripcion"] = (
        "Grafo de conocimiento del corpus ADL. Nodos: entidad, fragmento, "
        "documento. Aristas: CONTIENE, MENCIONA, RELACION."
    )

    por_chunk = {f["chunk_id"]: f for f in fragmentos}
    documentos_vistos: set[str] = set()

    for clave, entidad in entidades.items():
        grafo.add_node(
            f"E:{clave}",
            **_limpiar(
                {
                    "tipo_nodo": "entidad",
                    "nombre": entidad.nombre,
                    "tipo": entidad.tipo,
                    # Los alias son varios y GraphML no admite listas: van como
                    # cadena separada por barras verticales, que no aparece en
                    # ningún nombre normalizado porque `normalizar_nombre` deja
                    # solo caracteres alfanuméricos y espacios.
                    "alias": "|".join(entidad.alias),
                    "frecuencia_documental": entidad.frecuencia_documental,
                }
            ),
        )

    for chunk_id, menciones in menciones_por_chunk.items():
        fragmento = por_chunk.get(chunk_id)
        if fragmento is None:
            continue

        # Cuántas veces menciona este fragmento a cada entidad superviviente.
        conteo: dict[str, int] = {}
        for mencion in menciones:
            destino = clave_a_id.get(mencion.clave, mencion.clave)
            if f"E:{destino}" in grafo:
                conteo[destino] = conteo.get(destino, 0) + 1
        if not conteo:
            continue

        # Tope por fragmento: se quedan las más mencionadas. El desempate
        # alfabético mantiene el grafo reproducible entre corridas.
        elegidas = sorted(conteo, key=lambda c: (-conteo[c], c))[:max_por_fragmento]

        doc_id = fragmento["doc_id"]
        grafo.add_node(
            f"C:{chunk_id}",
            **_limpiar(
                {
                    "tipo_nodo": "fragmento",
                    "doc_id": doc_id,
                    "fuente": fragmento.get("fuente", ""),
                    "fenomeno": int(fragmento.get("fenomeno") or 0),
                    "posicion": int(fragmento.get("posicion") or 0),
                }
            ),
        )

        if doc_id not in documentos_vistos:
            documentos_vistos.add(doc_id)
            grafo.add_node(
                f"D:{doc_id}",
                **_limpiar(
                    {
                        "tipo_nodo": "documento",
                        "fuente": fragmento.get("fuente", ""),
                        "formato": fragmento.get("formato", ""),
                        "fenomeno": int(fragmento.get("fenomeno") or 0),
                    }
                ),
            )
        grafo.add_edge(f"D:{doc_id}", f"C:{chunk_id}", relacion="CONTIENE")

        for entidad_id in elegidas:
            grafo.add_edge(
                f"C:{chunk_id}",
                f"E:{entidad_id}",
                relacion="MENCIONA",
                n_menciones=conteo[entidad_id],
            )

    # Las relaciones van al final: para entonces se sabe qué entidades
    # sobrevivieron a la poda, y una tripleta cuyo sujeto u objeto no esté en el
    # grafo no se puede añadir sin inventar el nodo que falta.
    for tripleta in tripletas:
        sujeto, objeto = f"E:{tripleta.sujeto}", f"E:{tripleta.objeto}"
        if sujeto not in grafo or objeto not in grafo:
            continue
        if grafo.has_edge(sujeto, objeto):
            # Misma relación vista en otro fragmento: sube el conteo y se conserva
            # la primera evidencia. Repetirse en documentos distintos es la mejor
            # señal de que la tripleta no es un artefacto del patrón.
            grafo[sujeto][objeto]["apariciones"] += 1
            continue
        grafo.add_edge(
            sujeto,
            objeto,
            **_limpiar(
                {
                    "relacion": "RELACION",
                    "tipo": tripleta.tipo,
                    "doc_id": tripleta.doc_id,
                    "chunk_id": tripleta.chunk_id,
                    "evidencia": tripleta.evidencia,
                    "apariciones": 1,
                }
            ),
        )

    return grafo


def exportar(grafo, destino: Path) -> Path:
    """Escribe `grafo.graphml`, validando el esquema antes de tocar el disco."""
    import networkx as nx

    problemas = validar_escalares(grafo)
    if problemas:
        muestra = "\n  ".join(problemas[:5])
        raise TypeError(
            f"{len(problemas)} atributos no serializables a GraphML:\n  {muestra}"
        )

    destino.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(grafo, destino, encoding="utf-8", prettyprint=False)
    return destino


def resumen(grafo) -> dict[str, int]:
    """Conteos para el informe y para vigilar que la poda no se pase de lista."""
    por_nodo: dict[str, int] = {}
    for _, datos in grafo.nodes(data=True):
        clave = datos.get("tipo_nodo", "?")
        por_nodo[clave] = por_nodo.get(clave, 0) + 1

    por_arista: dict[str, int] = {}
    for _, _, datos in grafo.edges(data=True):
        clave = datos.get("relacion", "?")
        por_arista[clave] = por_arista.get(clave, 0) + 1

    return {
        "entidades": por_nodo.get("entidad", 0),
        "fragmentos": por_nodo.get("fragmento", 0),
        "documentos": por_nodo.get("documento", 0),
        "contiene": por_arista.get("CONTIENE", 0),
        "menciona": por_arista.get("MENCIONA", 0),
        "relaciones": por_arista.get("RELACION", 0),
    }
