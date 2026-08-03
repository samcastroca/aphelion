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
                )
                acumulado[chunk_id] = candidato
            candidato.puntaje += similitud
            candidato.posiciones[encoder] = posicion

    return sorted(acumulado.values(), key=lambda c: c.puntaje, reverse=True)


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


def agregar_a_documentos(
    candidatos: list[Candidato],
    top: int = config.TOP_DOCUMENTOS,
) -> list[str]:
    """Max pooling: cada documento hereda la puntuación de su mejor fragmento.

    Se descarta sum pooling por su sesgo de longitud: un documento con cuarenta
    fragmentos débiles acumularía más que un informe corto que contiene la
    respuesta exacta. Dada la heterogeneidad del corpus —alertas de un párrafo
    junto a atlas de cientos de páginas— ese sesgo sería severo.

    El número de fragmentos recuperados actúa solo como desempate, con peso
    suficientemente pequeño para no alterar el orden principal.
    """
    mejor: dict[str, float] = {}
    cuenta: dict[str, int] = {}

    for candidato in candidatos:
        anterior = mejor.get(candidato.doc_id)
        if anterior is None or candidato.puntaje > anterior:
            mejor[candidato.doc_id] = candidato.puntaje
        cuenta[candidato.doc_id] = cuenta.get(candidato.doc_id, 0) + 1

    ordenados = sorted(
        mejor.items(),
        key=lambda kv: (kv[1], cuenta[kv[0]] * 1e-9),
        reverse=True,
    )
    return [doc_id for doc_id, _ in ordenados[:top]]


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
    ):
        if not indices:
            raise ValueError("no hay índices cargados")
        self.indices = indices
        self.k0 = k0
        self.boost = boost
        self.max_por_doc = max_por_doc
        self.candidatos_por_indice = candidatos_por_indice
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
    ) -> Resultado:
        """La parte barata: fusionar, realzar, diversificar y agregar."""
        if fusion == "combsum":
            candidatos = fusionar_combsum(rankings)
        else:
            candidatos = fusionar_rrf(rankings, self.k0)
        candidatos = aplicar_boost_fenomeno(candidatos, consulta.fenomeno, self.boost)

        # La deduplicación se aplica solo a la lista de fragmentos. La agregación
        # a documento usa el conjunto completo: dos documentos distintos pueden
        # compartir un texto, y descartar uno lo volvería inalcanzable.
        fragmentos = diversificar(
            sin_duplicados(candidatos), top_fragmentos, self.max_por_doc
        )
        documentos = agregar_a_documentos(candidatos, top_documentos)

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
