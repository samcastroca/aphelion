"""Métricas de evaluación, transcritas literalmente de las fórmulas del reto.

Dos matices del enunciado que cambian el resultado si se pasan por alto:

- El recall de F1@3 se normaliza por `min(|Dq|, 3)`, no por `|Dq|`. Sin eso, una
  consulta con cinco documentos relevantes tendría techo 0.6 aunque el sistema
  acierte los tres que puede devolver.
- La relevancia de un fragmento se juzga por su contenido textual, no por su
  `chunk_id`. Como cada equipo aplica su propia estrategia de chunking,
  aquí el emparejamiento se hace contra el ground truth interno por `chunk_id`
  únicamente porque ese ground truth se construyó sobre estos mismos fragmentos.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Juicio:
    """Relevancias anotadas para una consulta."""

    query_id: str
    fragmentos: dict[str, float]  # chunk_id -> grado de relevancia (>= 0)
    documentos: set[str]  # doc_id relevantes

    @classmethod
    def from_dict(cls, d: dict) -> "Juicio":
        return cls(
            query_id=d["query_id"],
            fragmentos={k: float(v) for k, v in (d.get("fragmentos") or {}).items()},
            documentos=set(d.get("documentos") or []),
        )


def dcg(relevancias: list[float]) -> float:
    return sum(r / math.log2(i + 1) for i, r in enumerate(relevancias, start=1))


def ndcg_at_k(relevancias_obtenidas: list[float], ideal: list[float], k: int = 10) -> float:
    """NDCG@k. Devuelve 0.0 si no existe ningún documento relevante."""
    obtenido = dcg(relevancias_obtenidas[:k])
    maximo = dcg(sorted(ideal, reverse=True)[:k])
    return obtenido / maximo if maximo > 0 else 0.0


def f1_at_k(devueltos: list[str], relevantes: set[str], k: int = 3) -> float:
    """F1@k como métrica de conjunto: el orden no interviene."""
    if not relevantes:
        return 0.0

    seleccion = devueltos[:k]
    aciertos = len(set(seleccion) & relevantes)

    precision = aciertos / k
    recall = aciertos / min(len(relevantes), k)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluar(
    resultados: list[dict],
    juicios: dict[str, Juicio],
    k_fragmentos: int = 10,
    k_documentos: int = 3,
) -> dict:
    """Evalúa un `resultados.jsonl` contra el ground truth interno."""
    ndcgs: list[float] = []
    f1s: list[float] = []
    por_consulta: dict[str, dict] = {}

    for resultado in resultados:
        query_id = resultado["query_id"]
        juicio = juicios.get(query_id)
        if juicio is None:
            continue  # sin anotación no se puede puntuar

        obtenidas = [
            juicio.fragmentos.get(f["chunk_id"], 0.0)
            for f in resultado.get("fragments", [])
        ]
        ideal = list(juicio.fragmentos.values())
        ndcg = ndcg_at_k(obtenidas, ideal, k_fragmentos)

        devueltos = [d["doc_id"] for d in resultado.get("documents", [])]
        f1 = f1_at_k(devueltos, juicio.documentos, k_documentos)

        ndcgs.append(ndcg)
        f1s.append(f1)
        por_consulta[query_id] = {"ndcg@10": round(ndcg, 4), "f1@3": round(f1, 4)}

    return {
        "consultas_evaluadas": len(ndcgs),
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "f1@3": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "ic_ndcg@10": tuple(round(v, 4) for v in intervalo_bootstrap(ndcgs)),
        "ic_f1@3": tuple(round(v, 4) for v in intervalo_bootstrap(f1s)),
        "por_consulta": por_consulta,
    }


def evaluar_textual(
    resultados: list[dict],
    juicios: dict[str, Juicio],
    emparejadores: dict,
    mapa_fuente: dict[str, str] | None = None,
    k_fragmentos: int = 10,
    k_documentos: int = 3,
) -> dict:
    """Como `evaluar`, pero los fragmentos se emparejan por **texto**.

    Es la versión que hace falta para comparar configuraciones con distinta
    fragmentación: los `chunk_id` de un chunking de 256 tokens no existen en el
    ground truth, que se anotó sobre 504. Emparejar por texto es además lo que
    hará el jurado (§10.2.1).

    Los documentos se resuelven a `fuente` si se pasa el mapa, por la misma razón:
    es la clave con la que empareja el jurado, y el corpus tiene 59 nombres
    repetidos en 186 archivos.
    """
    ndcgs: list[float] = []
    f1s: list[float] = []
    por_consulta: dict[str, dict] = {}

    def resolver(doc_id: str) -> str:
        return mapa_fuente.get(doc_id, doc_id) if mapa_fuente else doc_id

    for resultado in resultados:
        query_id = resultado["query_id"]
        juicio = juicios.get(query_id)
        emparejador = emparejadores.get(query_id)
        if juicio is None or emparejador is None:
            continue

        obtenidas = emparejador.relevancias(
            [f["text"] for f in resultado.get("fragments", [])]
        )
        ndcg = ndcg_at_k(obtenidas, emparejador.ideal, k_fragmentos)

        devueltos = [resolver(d["doc_id"]) for d in resultado.get("documents", [])]
        relevantes = {resolver(d) for d in juicio.documentos}
        f1 = f1_at_k(devueltos, relevantes, k_documentos)

        ndcgs.append(ndcg)
        f1s.append(f1)
        por_consulta[query_id] = {"ndcg@10": round(ndcg, 4), "f1@3": round(f1, 4)}

    return {
        "consultas_evaluadas": len(ndcgs),
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "f1@3": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "ic_ndcg@10": tuple(round(v, 4) for v in intervalo_bootstrap(ndcgs)),
        "ic_f1@3": tuple(round(v, 4) for v in intervalo_bootstrap(f1s)),
        "por_consulta": por_consulta,
    }


def intervalo_bootstrap(
    valores: list[float],
    remuestreos: int = 2000,
    confianza: float = 0.95,
    semilla: int = 20260801,
) -> tuple[float, float]:
    """Intervalo de confianza de la media por bootstrap sobre las consultas.

    Con 50 consultas, una diferencia de NDCG@10 por debajo de 0.01 entra dentro
    del ruido y no distingue dos configuraciones. Sin este intervalo es fácil
    quedarse con la variante que ganó por azar en el barrido.
    """
    if not valores:
        return (0.0, 0.0)

    rng = random.Random(semilla)
    n = len(valores)
    medias = []
    for _ in range(remuestreos):
        muestra = [valores[rng.randrange(n)] for _ in range(n)]
        medias.append(sum(muestra) / n)
    medias.sort()

    cola = (1 - confianza) / 2
    bajo = medias[int(cola * remuestreos)]
    alto = medias[min(int((1 - cola) * remuestreos), remuestreos - 1)]
    return (bajo, alto)


def p_valor_permutacion(
    a: list[float],
    b: list[float],
    n: int = 100_000,
    semilla: int = 20260801,
) -> tuple[float, float]:
    """Compara dos configuraciones consulta a consulta. Devuelve (diferencia, p).

    `a` y `b` son la misma métrica de las mismas consultas y **en el mismo
    orden**, medida con dos configuraciones. Eso es lo que permite parear: la
    dificultad de cada consulta afecta a las dos por igual y se cancela al
    restar, mientras que comparar dos `intervalo_bootstrap` independientes la
    cuenta como varianza en ambas y esconde diferencias que sí son reales.

    Bajo la hipótesis nula las dos configuraciones son intercambiables en cada
    consulta, así que cambiarle el signo a una diferencia es una reasignación
    válida. El p-valor es la fracción de reasignaciones que dan una diferencia
    media al menos tan extrema como la observada; con la corrección de
    continuidad nunca sale exactamente cero, que es una certeza que un test de
    remuestreo no puede dar.

    Se prefiere a la prueba de hipótesis por bootstrap porque esta última tiene
    sesgo documentado hacia p-valores pequeños con pocas consultas, y aquí son
    cincuenta. `intervalo_bootstrap` sigue siendo lo correcto para poner un
    intervalo alrededor de **una** medida; lo que no hace bien es comparar dos.

    Ojo con la multiplicidad: nueve recetas son treinta y seis comparaciones por
    métrica, y a 0.05 se espera casi una falsa alarma por métrica. Si se van a
    mirar todas, hay que corregir el umbral.
    """
    if len(a) != len(b):
        raise ValueError(
            f"se comparan {len(a)} consultas contra {len(b)}: el test es pareado "
            "y exige las mismas consultas en el mismo orden"
        )
    if not a:
        raise ValueError("sin consultas no hay nada que comparar")

    difs = [x - y for x, y in zip(a, b)]
    observada = sum(difs) / len(difs)

    rng = random.Random(semilla)
    extremas = 0
    for _ in range(n):
        suma = sum(d if rng.random() < 0.5 else -d for d in difs)
        # El margen absorbe el error de coma flotante: sin él, la reasignación
        # que reproduce la diferencia original puede quedar fuera por 1e-17 y el
        # p-valor de dos medidas idénticas no llega a 1.
        if abs(suma / len(difs)) >= abs(observada) - 1e-12:
            extremas += 1

    return observada, (extremas + 1) / (n + 1)


def cargar_juicios(ruta: Path) -> dict[str, Juicio]:
    juicios: dict[str, Juicio] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            juicio = Juicio.from_dict(json.loads(linea))
            juicios[juicio.query_id] = juicio
    return juicios


def cargar_resultados(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
