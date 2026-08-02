"""Evalúa `resultados.jsonl` contra el ground truth interno.

El ground truth oficial no es público, así que se construye uno propio sobre las
50 consultas reales. Este script calcula NDCG@10 y F1@3 con las fórmulas del
reto y desglosa por fenómeno, que es donde suelen verse las diferencias reales
entre configuraciones.

Uso:
    uv run python scripts/04_evaluar.py
    uv run python scripts/04_evaluar.py --resultados otra_corrida.jsonl --detalle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aphelion import config, metricas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", type=Path, default=config.RESULTADOS)
    ap.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH)
    ap.add_argument("--detalle", action="store_true", help="desglose por consulta")
    args = ap.parse_args()

    if not args.resultados.exists():
        print(f"no existe {args.resultados}. Ejecuta antes generador.py")
        return 1
    if not args.ground_truth.exists():
        print(f"no existe {args.ground_truth}.")
        print("Construye el ground truth con scripts/05_pool_anotacion.py y anótalo.")
        return 1

    resultados = metricas.cargar_resultados(args.resultados)
    juicios = metricas.cargar_juicios(args.ground_truth)

    resumen = metricas.evaluar(resultados, juicios)

    print(f"consultas evaluadas: {resumen['consultas_evaluadas']} de {len(resultados)}")
    ic_n, ic_f = resumen["ic_ndcg@10"], resumen["ic_f1@3"]
    print(f"\n  NDCG@10  {resumen['ndcg@10']:.4f}   IC 95% [{ic_n[0]:.4f}, {ic_n[1]:.4f}]")
    print(f"  F1@3     {resumen['f1@3']:.4f}   IC 95% [{ic_f[0]:.4f}, {ic_f[1]:.4f}]")
    print(
        f"\n  Con {resumen['consultas_evaluadas']} consultas, dos configuraciones cuyos\n"
        "  intervalos se solapan no son distinguibles: la diferencia es ruido."
    )

    # Desglose por fenómeno: un promedio global puede esconder que un fenómeno
    # entero está fallando.
    por_fenomeno: dict[int, list[tuple[float, float]]] = {}
    for query_id, valores in resumen["por_consulta"].items():
        fenomeno = config.fenomeno_de_consulta(query_id)
        if fenomeno:
            por_fenomeno.setdefault(fenomeno, []).append(
                (valores["ndcg@10"], valores["f1@3"])
            )

    if por_fenomeno:
        print("\n  por fenómeno:")
        for fenomeno in sorted(por_fenomeno):
            pares = por_fenomeno[fenomeno]
            ndcg = sum(p[0] for p in pares) / len(pares)
            f1 = sum(p[1] for p in pares) / len(pares)
            print(f"    F{fenomeno}  n={len(pares):>2}  NDCG@10 {ndcg:.4f}  F1@3 {f1:.4f}")

    if args.detalle:
        print("\n  por consulta:")
        for query_id in sorted(resumen["por_consulta"]):
            valores = resumen["por_consulta"][query_id]
            print(
                f"    {query_id}  NDCG@10 {valores['ndcg@10']:.4f}  F1@3 {valores['f1@3']:.4f}"
            )

    sin_anotar = len(resultados) - resumen["consultas_evaluadas"]
    if sin_anotar:
        print(f"\n  aviso: {sin_anotar} consultas sin anotación quedaron fuera del promedio")

    return 0


if __name__ == "__main__":
    sys.exit(main())
