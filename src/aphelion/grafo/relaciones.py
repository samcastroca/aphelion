"""Etapa 2 del grafo: extracción de relaciones por patrones lingüísticos.

**Por qué patrones y no un modelo.** La §7.2 admite tres vías —clasificadores de
relaciones, heurísticas de patrones o dependencias sintácticas— y las otras dos
salen caras por motivos que no son técnicos:

- Los extractores generativos (REBEL, mREBEL) producen la tripleta token a token.
  Son decoders, y la §4.2 con la §8.3 los prohíben en indexación y recuperación.
  mREBEL además es CC BY-NC-SA 4.0, así que está excluido dos veces.
- GLiREL, que sí clasifica en vez de generar, es también CC BY-NC-SA 4.0.
- Los parsers de dependencias de spaCy en español son GPL-3.0 (heredada de UD
  AnCora) y en portugués CC BY-SA 4.0. Stanza es mejor apuesta —librería
  Apache-2.0, modelos ODC-BY 1.0— pero arrastra un modelo por idioma y su
  ganancia sobre esto está sin medir.

Queda `knowledgator/gliner-relex-large-v1.0` (Apache-2.0, clasificación sobre
pares, no generación) como mejora posible; no declara soporte multilingüe, así
que entra al barrido como candidato y no como sustituto.

**Cómo funciona.** Dentro de cada oración, se miran los pares de menciones
consecutivas y se comprueba si el texto que las separa encaja con alguno de los
patrones del inventario. Consecutivas y no todas contra todas porque en «EE. UU.,
China y Rusia desarrollan sistemas autónomos» el par (EE. UU., sistemas) tiene
entre medias otras dos entidades y el verbo ya no le pertenece de forma
inequívoca; encadenar pares vecinos recupera la enumeración sin inventar aristas.

**La procedencia va en la tripleta.** La §7.2 lo exige: cada una conserva su
`doc_id` y su `chunk_id`, más el fragmento de texto que la justifica. Eso es lo
que permite rastrear la evidencia de una relación hasta el documento original, y
es también lo que hace la arista útil como puente hacia los chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .entidades import Mencion

# Un par de menciones separado por más de esto no está relacionado por el verbo
# que haya en medio: hay demasiada oración por el camino. En caracteres, porque
# es una medida de distancia superficial y no de contenido.
MAX_SEPARACION = 60

# Inventario cerrado de relaciones. Cada entrada es (tipo, patrón, invierte); con
# `invierte` en True el sujeto y el objeto se intercambian, que es lo que hace
# falta para la voz pasiva: «X está regulado por Y» es (Y, regula, X).
#
# **Las pasivas van primero y el orden importa.** Se devuelve el primer patrón
# que encaje, y «developed by» contiene «developed»: con las activas delante,
# toda pasiva saldría con el sujeto y el objeto cambiados de sitio.
#
# **Las formas verbales están escritas, no comodinadas.** `desarroll\w+` parece
# más corto y captura además «desarrollo», que es un sustantivo: en «el
# desarrollo de X en Y» produciría la tripleta (X, desarrolla, Y), que no está en
# ningún sitio del texto. Enumerar las conjugaciones es más largo y no inventa
# relaciones.
#
# Los patrones cubren los tres idiomas del corpus en una sola expresión en vez de
# tener un inventario por idioma: es más corto de mantener y funciona en los
# fragmentos mezclados —citas en inglés dentro de un informe en español— que en
# este corpus son habituales.
_PATRONES: tuple[tuple[str, str, bool], ...] = (
    # --- pasivas (invierten sujeto y objeto) ---
    ("desarrolla", r"desarrollad[oa]s? por|desenvolvid[oa]s? por|developed by", True),
    ("opera", r"operad[oa]s? por|operated by|gestionad[oa]s? por|managed by", True),
    ("regula", r"regulad[oa]s? por|regulated by|governed by|sujet[oa]s? a|subject to", True),
    ("financia", r"financiad[oa]s? por|funded by|financed by|financiad[oa]s? pel[oa]", True),
    ("produce", r"producid[oa]s? por|fabricad[oa]s? por|produced by|manufactured by", True),
    ("lanza", r"lanzad[oa]s? por|launched by|lançad[oa]s? por", True),
    # --- activas ---
    ("desarrolla", r"desarroll(?:a|an|ó|aron|ando)|desenvolve[m]?|desenvolveu|develops?|developing", False),
    ("opera", r"opera[n]?|operó|operates?|operating|gestiona[n]?|manages?", False),
    ("regula", r"regula[n]?|reguló|regulates?|regulating|governs?|prohíbe[n]?|prohibits?", False),
    ("financia", r"financia[n]?|financió|funds|finances|subvenciona[n]?", False),
    ("produce", r"produce[n]?|produjo|produces?|fabrica[n]?|manufactures?", False),
    ("lanza", r"lanza[n]?|lanzó|launches|launched|lança|lançou", False),
    ("colabora_con", r"colabora con|coopera con|cooperates? with|collaborates? with|in partnership with|junto con|em parceria com|acuerdo con|agreement with", False),
    ("miembro_de", r"miembro de|member of|membro de|forma parte de|part of|integra", False),
    ("ubicado_en", r"con sede en|headquartered in|sediad[oa] em|ubicad[oa]s? en|located in|situad[oa]s? en|based in", False),
    ("firma", r"firma[n]?|firmó|signs?|signed|suscribe[n]?|suscribió|assina|assinou", False),
)

# Se compila una vez: esto se llama una vez por oración de 64.484 fragmentos.
_INVENTARIO: tuple[tuple[str, re.Pattern, bool], ...] = tuple(
    (tipo, re.compile(rf"\b(?:{patron})\b", re.IGNORECASE | re.UNICODE), invierte)
    for tipo, patron, invierte in _PATRONES
)

TIPOS_RELACION: tuple[str, ...] = tuple(dict.fromkeys(t for t, _, _ in _PATRONES))


@dataclass(frozen=True)
class Tripleta:
    """(sujeto, relación, objeto) con la procedencia que exige la §7.2."""

    sujeto: str  # clave normalizada de la entidad
    tipo: str
    objeto: str
    doc_id: str
    chunk_id: str
    evidencia: str  # el texto que justifica la relación, recortado

    @property
    def clave(self) -> tuple[str, str, str]:
        return (self.sujeto, self.tipo, self.objeto)


def _tramos_de_oracion(texto: str, oraciones: list[str]) -> list[tuple[int, int]]:
    """Devuelve el (inicio, fin) de cada oración dentro del texto original.

    `dividir_en_oraciones` entrega las oraciones ya recortadas, sin posiciones, y
    aquí hacen falta para saber qué menciones caen en la misma. Se localizan
    avanzando un cursor: como las oraciones salen del propio texto y en orden, la
    primera coincidencia a partir del cursor es siempre la correcta.
    """
    tramos: list[tuple[int, int]] = []
    cursor = 0
    for oracion in oraciones:
        if not oracion:
            continue
        inicio = texto.find(oracion, cursor)
        if inicio < 0:
            # Puede pasar si el segmentador normalizó espacios. Se salta esa
            # oración en vez de desalinear todas las siguientes.
            continue
        tramos.append((inicio, inicio + len(oracion)))
        cursor = inicio + len(oracion)
    return tramos


def _emparejar(hueco: str) -> tuple[str, bool] | None:
    """El primer patrón del inventario que encaje con el texto entre dos menciones.

    Se busca en cualquier posición del hueco y no solo al principio, porque entre
    el sujeto y su verbo cabe de todo —«El sistema, según el informe, está
    regulado por…»— y anclar al inicio pierde la mayoría de las pasivas. Lo que
    acota el falso positivo es `MAX_SEPARACION`, no la posición.

    El precio conocido es que la negación no se detecta: «Colombia nunca
    desarrolló X» produce la misma tripleta que la afirmativa. Es una limitación
    de la extracción por patrones, y la mitiga el conteo de `apariciones` del
    grafo: una tripleta que se repite en documentos distintos es difícil que sea
    un artefacto, y una que aparece una sola vez conviene leerla antes de creerla.
    """
    for tipo, patron, invierte in _INVENTARIO:
        if patron.search(hueco):
            return tipo, invierte
    return None


def extraer(
    texto: str,
    menciones: list[Mencion],
    oraciones: list[str],
    clave_a_id: dict[str, str] | None = None,
) -> list[Tripleta]:
    """Las tripletas de un fragmento.

    `clave_a_id` traduce la clave normalizada de cada mención al identificador de
    la entidad canónica; sin él, cada forma superficial sería su propio nodo y las
    aristas apuntarían a sitios que el grafo no tiene.
    """
    if len(menciones) < 2:
        return []

    clave_a_id = clave_a_id or {}
    ordenadas = sorted(menciones, key=lambda m: (m.inicio, m.fin))
    tramos = _tramos_de_oracion(texto, oraciones)
    if not tramos:
        tramos = [(0, len(texto))]

    tripletas: dict[tuple[str, str, str], Tripleta] = {}

    for inicio_o, fin_o in tramos:
        dentro = [m for m in ordenadas if m.inicio >= inicio_o and m.fin <= fin_o]
        for izquierda, derecha in zip(dentro, dentro[1:]):
            separacion = derecha.inicio - izquierda.fin
            if separacion <= 0 or separacion > MAX_SEPARACION:
                continue

            hueco = texto[izquierda.fin : derecha.inicio]
            encontrado = _emparejar(hueco)
            if encontrado is None:
                continue
            tipo, invierte = encontrado

            sujeto = clave_a_id.get(izquierda.clave, izquierda.clave)
            objeto = clave_a_id.get(derecha.clave, derecha.clave)
            if invierte:
                sujeto, objeto = objeto, sujeto
            if not sujeto or not objeto or sujeto == objeto:
                # Una entidad relacionada consigo misma aparece cuando dos formas
                # de la misma cosa caen en la misma oración («la NASA … la NASA»).
                # No es una relación, es una repetición.
                continue

            tripleta = Tripleta(
                sujeto=sujeto,
                tipo=tipo,
                objeto=objeto,
                doc_id=izquierda.doc_id,
                chunk_id=izquierda.chunk_id,
                evidencia=_recortar(texto, izquierda.inicio, derecha.fin),
            )
            # La misma tripleta puede salir dos veces del mismo fragmento; se
            # conserva una y el conteo de repeticiones se lleva en el grafo.
            tripletas.setdefault(tripleta.clave, tripleta)

    return list(tripletas.values())


# El texto que acompaña a la tripleta es evidencia, no contenido: entra en el
# GraphML como atributo y multiplicarlo por cada arista infla el archivo sin
# aportar. Con la oración recortada alrededor del par basta para verificarla a
# mano, que es para lo que sirve.
MAX_EVIDENCIA = 240


def _recortar(texto: str, inicio: int, fin: int) -> str:
    fragmento = " ".join(texto[inicio:fin].split())
    if len(fragmento) <= MAX_EVIDENCIA:
        return fragmento
    return fragmento[: MAX_EVIDENCIA - 1].rstrip() + "…"
