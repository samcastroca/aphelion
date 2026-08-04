"""Módulo de recuperación.

Ninguna etapa emplea modelos generativos: no hay reranking por LLM, expansión de
consulta, filtrado generativo ni síntesis. Todo opera sobre vectores, puntuaciones
de similitud y metadata.

El orden de operaciones es deliberado:

    búsqueda por índice -> fusión RRF -> boost por fenómeno -> diversificación
    -> top-10 fragmentos -> max pooling -> top-3 documentos

La diversificación va antes del pooling porque su propósito es proteger la lista
de fragmentos, que es lo que evalúa NDCG@10; la agregación a documento se calcula
sobre el conjunto completo de candidatos para no perder señal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from .consultas import Consulta
from ..indice.vectores import IndiceVectorial


@dataclass
class Candidato:
    chunk_id: str
    doc_id: str
    texto: str
    fuente: str
    fenomeno: int
    puntaje: float
    posiciones: dict[str, int]  # encoder -> rango en su índice (base 1)
    idioma: str = "es"  # del fragmento; decide el segmentador al recortar

    @property
    def consenso(self) -> int:
        """Número de índices en los que el fragmento aparece."""
        return len(self.posiciones)


@dataclass
class Resultado:
    query_id: str
    documentos: list[str]  # doc_id, de mayor a menor relevancia
    fragmentos: list[Candidato]


# --- Fusión ---------------------------------------------------------------


def fusionar_rrf(
    rankings: dict[str, list[tuple[dict, float]]],
    k0: int = config.RRF_K0,
) -> list[Candidato]:
    """Reciprocal Rank Fusion sobre los rankings de cada encoder.

    RRF opera sobre posiciones y no sobre puntuaciones, lo que lo hace inmune a la
    diferencia de escalas entre espacios vectoriales distintos: la similitud
    coseno de BGE-M3 y la de E5 no son comparables entre sí, pero sus órdenes sí.
    """
    acumulado: dict[str, Candidato] = {}

    for encoder, ranking in rankings.items():
        for posicion, (meta, _similitud) in enumerate(ranking, start=1):
            chunk_id = meta["chunk_id"]
            candidato = acumulado.get(chunk_id)
            if candidato is None:
                candidato = Candidato(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    texto=meta["texto"],
                    fuente=meta["fuente"],
                    fenomeno=meta.get("fenomeno", 0),
                    puntaje=0.0,
                    posiciones={},
                    idioma=meta.get("idioma") or "es",
                )
                acumulado[chunk_id] = candidato
            candidato.puntaje += 1.0 / (k0 + posicion)
            candidato.posiciones[encoder] = posicion

    return sorted(acumulado.values(), key=lambda c: c.puntaje, reverse=True)


def fusionar_combsum(
    rankings: dict[str, list[tuple[dict, float]]],
) -> list[Candidato]:
    """Suma las similitudes de cada fragmento en los índices donde aparece.

    La alternativa a RRF. Conserva la magnitud de la similitud en lugar de
    reducirla a una posición, lo que en teoría aporta información: la diferencia
    entre coseno 0.9 y 0.5 se pierde al ordenar. Funciona aquí porque los dos
    encoders devuelven coseno en el mismo rango; con un ranker léxico al lado no
    sería comparable sin normalizar.

    Un fragmento ausente del top-k de un índice suma cero por ese índice.
    """
    acumulado: dict[str, Candidato] = {}

    for encoder, ranking in rankings.items():
        for posicion, (meta, similitud) in enumerate(ranking, start=1):
            chunk_id = meta["chunk_id"]
            candidato = acumulado.get(chunk_id)
            if candidato is None:
                candidato = Candidato(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    texto=meta["texto"],
                    fuente=meta["fuente"],
                    fenomeno=meta.get("fenomeno", 0),
                    puntaje=0.0,
                    posiciones={},
                    idioma=meta.get("idioma") or "es",
                )
                acumulado[chunk_id] = candidato
            candidato.puntaje += similitud
            candidato.posiciones[encoder] = posicion

    return sorted(acumulado.values(), key=lambda c: c.puntaje, reverse=True)


def fusionar_convexa(
    rankings: dict[str, list[tuple[dict, float]]],
    pesos: dict[str, float] | None = None,
) -> list[Candidato]:
    """Combinación convexa de similitudes normalizadas por consulta.

    La alternativa que la literatura sitúa por encima de RRF cuando hay
    etiquetas de relevancia en dominio (Bruch et al., 2022), y el diseño ya la
    tenía anotada como pendiente de medir. Frente a CombSUM crudo añade la
    normalización, que es lo que hace comparables dos escalas de coseno
    distintas; frente a RRF conserva la magnitud, que es la información que
    reducir a posiciones tira.

    La normalización es min-max **dentro de la consulta y del índice**: el mejor
    candidato de cada índice vale 1 y el peor 0. Eso hace que la escala absoluta
    de cada encoder deje de importar sin perder las distancias relativas.

    Con un solo índice el resultado es una reescala monótona de la similitud, así
    que el orden coincide con el de ese encoder.
    """
    acumulado: dict[str, Candidato] = {}

    for encoder, ranking in rankings.items():
        if not ranking:
            continue
        peso = (pesos or {}).get(encoder, 1.0)
        similitudes = [sim for _, sim in ranking]
        alto, bajo = max(similitudes), min(similitudes)
        rango = alto - bajo

        for posicion, (meta, similitud) in enumerate(ranking, start=1):
            # Si todos empatan, la normalización no está definida: se les da a
            # todos el mismo valor en vez de dividir por cero.
            normalizada = 1.0 if rango <= 0 else (similitud - bajo) / rango
            chunk_id = meta["chunk_id"]
            candidato = acumulado.get(chunk_id)
            if candidato is None:
                candidato = Candidato(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    texto=meta["texto"],
                    fuente=meta["fuente"],
                    fenomeno=meta.get("fenomeno", 0),
                    puntaje=0.0,
                    posiciones={},
                    idioma=meta.get("idioma") or "es",
                )
                acumulado[chunk_id] = candidato
            candidato.puntaje += peso * normalizada
            candidato.posiciones[encoder] = posicion

    return sorted(acumulado.values(), key=lambda c: c.puntaje, reverse=True)


def filtrar_por_umbral_relativo(
    ranking: list[tuple[dict, float]],
    umbral: float | None,
) -> list[tuple[dict, float]]:
    """Post-filtro de la §8.7: descarta la cola de baja similitud de un índice.

    El umbral es relativo al mejor candidato de esa consulta en ese índice, no
    absoluto, porque las escalas de coseno de los dos encoders no son
    comparables: mE5 comprime todo hacia 0,75–0,92 y BGE-M3 se mueve más abajo.
    Como el ranking viene ordenado de mayor a menor, el filtro equivale a un
    k adaptativo por consulta y no altera las posiciones de lo que conserva.
    """
    if not umbral or not ranking:
        return ranking

    tope = ranking[0][1]
    if tope <= 0:  # sin señal utilizable, mejor no filtrar nada
        return ranking

    corte = umbral * tope
    return [(meta, sim) for meta, sim in ranking if sim >= corte]


def aplicar_boost_fenomeno(
    candidatos: list[Candidato],
    fenomeno: int | None,
    factor: float = config.BOOST_FENOMENO,
) -> list[Candidato]:
    """Realza los candidatos del fenómeno esperado sin excluir a los demás.

    Se usa un multiplicador suave y no un filtro duro porque varias consultas
    admiten evidencia transversal: q005 y q046 tratan Colombia desde fenómenos
    distintos, y q027 cruza inteligencia artificial con operaciones espaciales.
    """
    if fenomeno is None or factor == 1.0:
        return candidatos

    for candidato in candidatos:
        if candidato.fenomeno == fenomeno:
            candidato.puntaje *= factor

    return sorted(candidatos, key=lambda c: c.puntaje, reverse=True)


def sin_duplicados(candidatos: list[Candidato]) -> list[Candidato]:
    """Elimina fragmentos con texto repetido, conservando el mejor puntuado.

    El corpus tiene 4.445 fragmentos (3%) con texto exactamente idéntico: mismas
    tablas en informes distintos, mismos encabezados institucionales. Dos copias
    del mismo texto en el top-10 tienen la misma relevancia y gastan dos de las
    diez posiciones que evalúa NDCG@10 para informar una sola vez.

    Se compara sobre el texto normalizado por espacios, de modo que también caen
    las variantes que solo difieren en el salto de línea.
    """
    vistos: set[str] = set()
    unicos: list[Candidato] = []

    for candidato in candidatos:
        clave = " ".join(candidato.texto.split()).casefold()
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append(candidato)

    return unicos


def diversificar(
    candidatos: list[Candidato],
    limite: int,
    max_por_doc: int = config.MAX_FRAGMENTOS_POR_DOC,
) -> list[Candidato]:
    """Limita cuántos fragmentos aporta cada documento a la lista final.

    Sin esta restricción un solo documento puede ocupar las diez posiciones
    evaluadas; si resulta no ser relevante, la consulta se pierde entera. El tope
    reparte el riesgo entre varios documentos.
    """
    seleccionados: list[Candidato] = []
    por_doc: dict[str, int] = {}

    for candidato in candidatos:
        if por_doc.get(candidato.doc_id, 0) >= max_por_doc:
            continue
        seleccionados.append(candidato)
        por_doc[candidato.doc_id] = por_doc.get(candidato.doc_id, 0) + 1
        if len(seleccionados) >= limite:
            break

    # Si el tope dejó la lista corta, se completa con los mejores descartados.
    if len(seleccionados) < limite:
        ya = {c.chunk_id for c in seleccionados}
        for candidato in candidatos:
            if candidato.chunk_id in ya:
                continue
            seleccionados.append(candidato)
            if len(seleccionados) >= limite:
                break

    return seleccionados


def _puntaje_documento(puntajes: list[float], modo: str) -> float:
    """Colapsa las puntuaciones de los fragmentos de un documento en una.

    - `max`: el mejor fragmento manda. Es el que se entrega; no tiene sesgo de
      longitud, pero descarta la evidencia repetida.
    - `suma`: premia acumular fragmentos y por eso tiene sesgo de longitud. Se
      incluye porque el diseño lo descartó por argumento y conviene medirlo.
    - `media`: quita el sesgo de longitud de la suma, pero castiga al documento
      largo que acierta en un párrafo y falla en los demás.
    - `topN`: media de los N mejores. El punto medio entre `max` y `suma`:
      recompensa que la evidencia esté concentrada en varios fragmentos buenos
      sin dejar que cuarenta fragmentos flojos ganen por acumulación.
    """
    if modo == "suma":
        return sum(puntajes)
    if modo == "media":
        return sum(puntajes) / len(puntajes)
    if modo.startswith("top"):
        n = int(modo[3:])
        mejores = sorted(puntajes, reverse=True)[:n]
        return sum(mejores) / len(mejores)
    return max(puntajes)  # 'max' y cualquier valor desconocido


def agregar_a_documentos(
    candidatos: list[Candidato],
    top: int = config.TOP_DOCUMENTOS,
    modo: str = "max",
) -> list[str]:
    """Max pooling: cada documento hereda la puntuación de su mejor fragmento.

    Se descarta sum pooling por su sesgo de longitud: un documento con cuarenta
    fragmentos débiles acumularía más que un informe corto que contiene la
    respuesta exacta. Dada la heterogeneidad del corpus —alertas de un párrafo
    junto a atlas de cientos de páginas— ese sesgo sería severo.

    La agregación es por `fuente` y no por `doc_id`, porque el jurado empareja
    los documentos por `fuente` (§10.2.1) y el corpus tiene 59 nombres
    estandarizados repetidos en 186 archivos: dos doc_id con la misma fuente en
    el top-3 cuentan como un solo acierto posible y desperdician una posición.
    De cada fuente se reporta el doc_id de su mejor fragmento.

    El número de fragmentos recuperados actúa solo como desempate, con peso
    suficientemente pequeño para no alterar el orden principal.
    """
    puntajes: dict[str, list[float]] = {}
    mejor: dict[str, float] = {}
    representante: dict[str, str] = {}  # fuente -> doc_id del mejor fragmento

    for candidato in candidatos:
        clave = candidato.fuente or candidato.doc_id
        puntajes.setdefault(clave, []).append(candidato.puntaje)
        anterior = mejor.get(clave)
        if anterior is None or candidato.puntaje > anterior:
            mejor[clave] = candidato.puntaje
            representante[clave] = candidato.doc_id

    agregado = {clave: _puntaje_documento(v, modo) for clave, v in puntajes.items()}
    ordenados = sorted(
        agregado.items(),
        key=lambda kv: (kv[1], len(puntajes[kv[0]]) * 1e-9),
        reverse=True,
    )
    return [representante[clave] for clave, _ in ordenados[:top]]


# --- Recuperador ----------------------------------------------------------


class Recuperador:
    """Encapsula los índices cargados y la política de recuperación."""

    def __init__(
        self,
        indices: dict[str, IndiceVectorial],
        encoders_cargados: dict | None = None,
        k0: int = config.RRF_K0,
        boost: float = config.BOOST_FENOMENO,
        max_por_doc: int = config.MAX_FRAGMENTOS_POR_DOC,
        candidatos_por_indice: int = config.CANDIDATOS_POR_INDICE,
        umbral_relativo: float | None = config.UMBRAL_RELATIVO,
    ):
        if not indices:
            raise ValueError("no hay índices cargados")
        self.indices = indices
        self.k0 = k0
        self.boost = boost
        self.max_por_doc = max_por_doc
        self.candidatos_por_indice = candidatos_por_indice
        self.umbral_relativo = umbral_relativo
        self._encoders = encoders_cargados or {}

    def _encoder(self, nombre: str):
        if nombre not in self._encoders:
            from ..indice import encoders

            self._encoders[nombre] = encoders.cargar(nombre)
        return self._encoders[nombre]

    def buscar(
        self, consulta: Consulta, candidatos_por_indice: int | None = None
    ) -> dict[str, list[tuple[dict, float]]]:
        """La parte cara: codificar la consulta y buscar en cada índice.

        Está separada del resto porque no depende de ningún hiperparámetro de
        fusión. El barrido busca una vez y reordena cien veces.
        """
        k = candidatos_por_indice or self.candidatos_por_indice
        rankings: dict[str, list[tuple[dict, float]]] = {}
        for nombre, indice in self.indices.items():
            vector = self._encoder(nombre).codificar_consultas([consulta.texto])[0]
            rankings[nombre] = [
                (indice.meta(pos), sim) for pos, sim in indice.buscar(vector, k)
            ]
        return rankings

    def ordenar(
        self,
        rankings: dict[str, list[tuple[dict, float]]],
        consulta: Consulta,
        top_fragmentos: int = config.TOP_FRAGMENTOS,
        top_documentos: int = config.TOP_DOCUMENTOS,
        fusion: str = "rrf",
        agregacion: str = "max",
    ) -> Resultado:
        """La parte barata: fusionar, realzar, diversificar y agregar."""
        # El filtro va antes de la fusión y dentro de esta etapa, no en `buscar`:
        # así el barrido puede variar el umbral sobre el mismo pool cacheado.
        if self.umbral_relativo:
            rankings = {
                nombre: filtrar_por_umbral_relativo(lista, self.umbral_relativo)
                for nombre, lista in rankings.items()
            }
        if fusion == "combsum":
            candidatos = fusionar_combsum(rankings)
        elif fusion == "convexa":
            candidatos = fusionar_convexa(rankings)
        else:
            candidatos = fusionar_rrf(rankings, self.k0)
        candidatos = aplicar_boost_fenomeno(candidatos, consulta.fenomeno, self.boost)

        # La deduplicación se aplica solo a la lista de fragmentos. La agregación
        # a documento usa el conjunto completo: dos documentos distintos pueden
        # compartir un texto, y descartar uno lo volvería inalcanzable.
        #
        # Se entregan más candidatos de los que se publican: la construcción de
        # la salida puede descartar piezas por el tope por documento, y con
        # repuestos reales a mano no tiene que rellenar repitiendo fragmentos.
        fragmentos = diversificar(
            sin_duplicados(candidatos), top_fragmentos * 3, self.max_por_doc
        )
        documentos = agregar_a_documentos(candidatos, top_documentos, agregacion)

        return Resultado(consulta.query_id, documentos, fragmentos)

    def recuperar(
        self,
        consulta: Consulta,
        top_fragmentos: int = config.TOP_FRAGMENTOS,
        top_documentos: int = config.TOP_DOCUMENTOS,
    ) -> Resultado:
        rankings = self.buscar(consulta)
        return self.ordenar(rankings, consulta, top_fragmentos, top_documentos)


def cargar_indices(
    nombres: list[str] | None = None, base=None
) -> dict[str, IndiceVectorial]:
    from ..indice import vectores

    disponibles = vectores.encoders_disponibles(base)
    if not disponibles:
        raise FileNotFoundError(
            "no hay índices construidos; ejecuta antes scripts/etapas/04_indexar.py"
        )

    elegidos = nombres or disponibles
    faltantes = set(elegidos) - set(disponibles)
    if faltantes:
        raise FileNotFoundError(f"índices no encontrados: {sorted(faltantes)}")

    return {nombre: vectores.cargar(nombre, base) for nombre in elegidos}
