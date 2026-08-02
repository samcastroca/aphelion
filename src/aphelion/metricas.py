"""Métricas de evaluación, réplica de las fórmulas de la §10.2.

Dos matices del enunciado que cambian el resultado si se pasan por alto:

- El recall de F1@3 se normaliza por `min(|Dq|, 3)`, no por `|Dq|`. Sin eso, una
  consulta con cinco documentos relevantes tendría techo 0.6 aunque el sistema
  acierte los tres que puede devolver.
- La relevancia de un fragmento se juzga por su contenido textual, no por su
  `chunk_id` (§10.2.1). Como cada equipo aplica su propia estrategia de chunking,
  aquí el emparejamiento se hace contra el ground truth interno por `chunk_id`
  únicamente porque ese ground truth se construyó sobre estos mismos fragmentos.
"""

from __future__ import annotations

import json
import math
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
        "por_consulta": por_consulta,
    }


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
