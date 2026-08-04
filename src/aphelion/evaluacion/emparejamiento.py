"""Emparejamiento de fragmentos por texto, no por `chunk_id`.

**Por qué existe.** El ground truth interno se anotó sobre una fragmentación
concreta: 504 tokens con 15% de solape. Sus juicios están indexados por
`chunk_id`, y un `chunk_id` solo significa algo dentro de la fragmentación que lo
produjo. En cuanto el barrido prueba otro tamaño de chunk, `F3-SIPRI-111-chunk-0023`
deja de existir —o peor, existe y describe un texto distinto— y evaluar por
identificador daría un NDCG@10 sin sentido, no un cero honesto.

Emparejar por texto resuelve eso y además es **lo que hará el jurado**: la
§10.2.1 dice que la relevancia de un fragmento se juzga sobre su contenido
textual y que el `chunk_id` es solo trazabilidad. Así que este módulo no es un
apaño para el barrido: acerca la medición interna a la oficial.

**Cómo empareja.** Sobre bolsas de n-gramas de palabras normalizadas, y con
contención en el sentido más favorable de los dos:

    solape(R, J) = max( |R∩J| / |R| , |R∩J| / |J| )

Los dos sentidos hacen falta porque el tamaño de chunk cambia en ambas
direcciones. Con chunks de 768 tokens, el fragmento devuelto R contiene entero al
juzgado J y lo que mide es |R∩J|/|J|; con chunks de 256, R cae dentro de J y lo
que mide es |R∩J|/|R|. Un solo sentido declararía irrelevante la mitad de los
casos.

Un fragmento devuelto hereda **el grado máximo** de los juicios con los que
solapa por encima del umbral. Es deliberado: si R contiene el pasaje que responde
la consulta, R responde la consulta, aunque venga con contexto alrededor.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# 5-gramas de palabras. Menos que eso empareja por coincidencias casuales de
# vocabulario —dos fragmentos del mismo informe comparten muchas palabras
# sueltas—; más que eso se vuelve frágil ante una palabra distinta en el medio,
# que es justo lo que introduce recortar por otra frontera oracional.
N = 5

# Calibrado, no elegido a ojo. Se midió por las dos caras sobre este corpus:
#
# - **Falsos negativos.** Re-fragmentando la submuestra y buscando el equivalente
#   por texto de cada uno de los 957 fragmentos juzgados, con 0.60 quedaban 29
#   huérfanos y los 29 caían entre 0.588 y 0.599: un grupo pegado al umbral, no
#   una cola de casos genuinamente distintos. Con 0.55 se recuperan todos.
# - **Falsos positivos.** Sobre la *misma* fragmentación que originó el ground
#   truth, emparejar por texto debe dar lo mismo que emparejar por `chunk_id`.
#   Da 0.7304 frente a 0.7280 de NDCG@10 con 0.60 y 0.7328 con 0.55: una décima
#   parte del intervalo de confianza, así que bajar el umbral no está inventando
#   relevancia.
#
# Por debajo de 0.5 sí empiezan a colar vecinos que solo comparten el arranque
# por el solape del 15%; por encima de 0.75 se pierden emparejamientos legítimos
# en cuanto el recorte a 250 palabras corta el fragmento por otro sitio.
UMBRAL = 0.55

_NO_ALFANUM = re.compile(r"[^\w\s]+", re.UNICODE)
_ESPACIOS = re.compile(r"\s+")


def normalizar(texto: str) -> str:
    """Minúsculas sin tildes ni puntuación: el emparejamiento no debe depender de
    si un lado conservó una coma o una ligadura tipográfica."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = _NO_ALFANUM.sub(" ", texto)
    return _ESPACIOS.sub(" ", texto).strip()


def ngramas(texto: str, n: int = N) -> frozenset[tuple[str, ...]]:
    palabras = normalizar(texto).split()
    if len(palabras) < n:
        # Un fragmento más corto que la ventana se representa por sí mismo, o
        # quedaría con bolsa vacía y no emparejaría nunca. Ocurre con los
        # fragmentos de solo título de los artículos del CEEEP.
        return frozenset({tuple(palabras)}) if palabras else frozenset()
    return frozenset(tuple(palabras[i : i + n]) for i in range(len(palabras) - n + 1))


def solape(a: frozenset, b: frozenset) -> float:
    """Contención en el sentido más favorable. 0.0 si alguna bolsa está vacía."""
    if not a or not b:
        return 0.0
    comun = len(a & b)
    if not comun:
        return 0.0
    return max(comun / len(a), comun / len(b))


@dataclass
class JuicioTextual:
    """Un fragmento juzgado, listo para emparejar."""

    chunk_id: str
    doc_id: str
    relevancia: float
    bolsa: frozenset


class Emparejador:
    """Asigna relevancias a fragmentos devueltos, emparejándolos por texto.

    Se construye una vez por consulta y se consulta por cada fragmento devuelto;
    las bolsas de los juicios se calculan una sola vez, que es lo caro.
    """

    def __init__(self, juicios: list[JuicioTextual], umbral: float = UMBRAL):
        self.juicios = juicios
        self.umbral = umbral

    @property
    def ideal(self) -> list[float]:
        """Las relevancias anotadas, para el IDCG."""
        return [j.relevancia for j in self.juicios]

    def relevancia(self, texto: str) -> float:
        """Grado del fragmento devuelto: el máximo de los juicios con que solapa."""
        bolsa = ngramas(texto)
        if not bolsa:
            return 0.0
        mejor = 0.0
        for juicio in self.juicios:
            if juicio.relevancia <= mejor:
                continue  # no puede mejorar lo que ya tenemos
            if solape(bolsa, juicio.bolsa) >= self.umbral:
                mejor = juicio.relevancia
        return mejor

    def relevancias(self, textos: list[str]) -> list[float]:
        return [self.relevancia(t) for t in textos]


def cargar_textos(ruta: Path) -> dict[str, str]:
    """chunk_id -> texto, de `data/ground_truth_textos.jsonl`."""
    textos: dict[str, str] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            textos[registro["chunk_id"]] = registro["texto"]
    return textos


def emparejadores(
    juicios_por_consulta: dict,
    textos: dict[str, str],
    umbral: float = UMBRAL,
) -> dict[str, Emparejador]:
    """Un `Emparejador` por consulta, a partir del ground truth y los textos.

    Los juicios sin texto conocido se descartan con aviso: emparejar por texto
    exige el texto, y silenciarlo convertiría un juicio perdido en un cero.
    """
    salida: dict[str, Emparejador] = {}
    sin_texto = 0

    for query_id, juicio in juicios_por_consulta.items():
        lista: list[JuicioTextual] = []
        for chunk_id, grado in juicio.fragmentos.items():
            texto = textos.get(chunk_id)
            if texto is None:
                sin_texto += 1
                continue
            lista.append(
                JuicioTextual(
                    chunk_id=chunk_id,
                    doc_id=chunk_id.rsplit("-chunk-", 1)[0],
                    relevancia=grado,
                    bolsa=ngramas(texto),
                )
            )
        salida[query_id] = Emparejador(lista, umbral)

    if sin_texto:
        print(
            f"  aviso: {sin_texto} fragmentos juzgados sin texto en "
            f"ground_truth_textos.jsonl; no podrán emparejarse"
        )
    return salida
