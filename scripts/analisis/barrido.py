"""Prueba muchas configuraciones de recuperación y dice cuál gana.

La idea que lo hace viable: buscar en los índices es lo caro y no depende de
ningún hiperparámetro de fusión. Así que se busca **una sola vez** por consulta,
con el pool más profundo del barrido, y después cada configuración reordena esos
mismos candidatos. Cien configuraciones cuestan lo que una.

Lo que sí exige es ground truth. Sin anotaciones no hay «mejor»: hay salidas
distintas y ninguna forma de compararlas. Si falta, el script lo dice y para.

Las configuraciones se ordenan como el reto ordena a los equipos: Conteo de
Borda sobre las dos métricas, que las pondera por igual. Y no se declara ganador
si los intervalos de confianza se solapan, porque con 50 consultas una
diferencia por debajo de 0,01 es ruido.

Uso:
    uv run python scripts/analisis/barrido.py
    uv run python scripts/analisis/barrido.py --rapido      # rejilla reducida
    uv run python scripts/analisis/barrido.py --aplicar     # escribe el ganador a config
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas, recuperacion, salida
from aphelion.evaluacion import metricas

# Rejilla completa. Todos estos valores se defendieron a priori en el diseño y
# ninguno se ha medido todavía contra datos.
#
# `umbral` es el post-filtro relativo de la §8.7: None lo desactiva; 0.9 exige
# al menos el 90% de la similitud del mejor candidato de ese índice. Se barre
# aquí y no se fija a priori porque su efecto depende de cuánta cola de ruido
# traiga cada consulta, que es exactamente lo que no se sabe sin ground truth.
REJILLA = {
    "fusion": ["rrf", "combsum"],
    "k0": [10, 20, 60],
    "boost": [1.0, 1.05, 1.15],
    "max_por_doc": [2, 3, 5],
    "candidatos": [100, 200],
    "subdividir": [False, True],
    "umbral": [None, 0.9],
}

REJILLA_RAPIDA = {
    "fusion": ["rrf", "combsum"],
    "k0": [20, 60],
    "boost": [1.0, 1.05],
    "max_por_doc": [3],
    "candidatos": [200],
    "subdividir": [False],
    "umbral": [None],
}


@dataclass
class Corrida:
    cfg: dict
    ndcg: float
    f1: float
    ic_ndcg: tuple[float, float]
    ic_f1: tuple[float, float]

    @property
    def etiqueta(self) -> str:
        umbral = self.cfg.get("umbral")
        return (
            f"{self.cfg['fusion']:<8} k0={self.cfg['k0']:<3} "
            f"boost={self.cfg['boost']:<5} doc={self.cfg['max_por_doc']} "
            f"cand={self.cfg['candidatos']:<4} sub={'sí' if self.cfg['subdividir'] else 'no'} "
            f"umbral={umbral if umbral else 'no'}"
        )


def combinaciones(rejilla: dict) -> list[dict]:
    claves = list(rejilla)
    return [dict(zip(claves, valores)) for valores in itertools.product(*rejilla.values())]


def puntos_borda(corridas: list[Corrida]) -> dict[int, int]:
    """Conteo de Borda sobre las dos métricas, que es como se ordena el reto.

    A la configuración en la posición p de una tabla le corresponden N - p
    puntos. Sumar las dos tablas premia el desempeño consistente en ambas, no
    ganar una a costa de la otra.
    """
    n = len(corridas)
    puntos = {i: 0 for i in range(n)}

    for clave in (lambda c: c.ndcg, lambda c: c.f1):
        orden = sorted(range(n), key=lambda i: clave(corridas[i]), reverse=True)
        for posicion, i in enumerate(orden, start=1):
            puntos[i] += n - posicion

    return puntos


def indistinguibles(a: Corrida, b: Corrida) -> bool:
    """Dos configuraciones cuyos intervalos se solapan en ambas métricas."""
    solapa_ndcg = a.ic_ndcg[0] <= b.ic_ndcg[1] and b.ic_ndcg[0] <= a.ic_ndcg[1]
    solapa_f1 = a.ic_f1[0] <= b.ic_f1[1] and b.ic_f1[0] <= a.ic_f1[1]
    return solapa_ndcg and solapa_f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="rejilla reducida")
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH)
    ap.add_argument("--salida", type=Path, default=config.TRABAJO / "barrido.jsonl")
    ap.add_argument("--top", type=int, default=15, help="cuántas configuraciones mostrar")
    args = ap.parse_args()

    if not args.ground_truth.exists():
        print(f"No existe {args.ground_truth}.")
        print()
        print("Un barrido compara configuraciones contra anotaciones de relevancia.")
        print("Sin ellas no hay nada que comparar. El orden es:")
        print("  1. uv run python scripts/analisis/pool_anotacion.py")
        print("  2. el equipo rellena la columna 'relevancia' en los CSV")
        print("  3. uv run python scripts/analisis/pool_anotacion.py --consolidar")
        print("  4. este script")
        return 1

    juicios = metricas.cargar_juicios(args.ground_truth)
    preguntas = mod_consultas.cargar()
    anotadas = [p for p in preguntas if p.query_id in juicios]
    print(f"consultas con anotación: {len(anotadas)} de {len(preguntas)}")
    if len(anotadas) < 10:
        print("  muy pocas para comparar configuraciones con sentido")
        return 1

    rejilla = REJILLA_RAPIDA if args.rapido else REJILLA
    configs = combinaciones(rejilla)
    print(f"configuraciones a probar: {len(configs)}")

    indices = recuperacion.cargar_indices(None, args.base)
    for nombre, indice in indices.items():
        print(f"  índice {nombre}: {len(indice):,} fragmentos")

    # Una sola búsqueda por consulta, con el pool más profundo de la rejilla.
    profundidad = max(rejilla["candidatos"])
    buscador = recuperacion.Recuperador(indices)
    print(f"\nbuscando una vez por consulta (pool de {profundidad}) ...", flush=True)
    t0 = time.time()
    cache = {
        p.query_id: buscador.buscar(p, candidatos_por_indice=profundidad)
        for p in tqdm(anotadas, desc="consultas", unit="q")
    }
    print(f"  {time.time() - t0:.1f}s\n")

    corridas: list[Corrida] = []
    for cfg in tqdm(configs, desc="configuraciones", unit="cfg"):
        recuperador = recuperacion.Recuperador(
            indices,
            k0=cfg["k0"],
            boost=cfg["boost"],
            max_por_doc=cfg["max_por_doc"],
            umbral_relativo=cfg.get("umbral"),
        )
        resultados = []
        for pregunta in anotadas:
            # Recortar el pool cacheado equivale a haber buscado con ese k.
            rankings = {
                nombre: lista[: cfg["candidatos"]]
                for nombre, lista in cache[pregunta.query_id].items()
            }
            resultado = recuperador.ordenar(rankings, pregunta, fusion=cfg["fusion"])
            resultados.append(
                {
                    "query_id": resultado.query_id,
                    "documents": [
                        {"rank": i, "doc_id": d}
                        for i, d in enumerate(resultado.documentos, start=1)
                    ],
                    "fragments": salida.construir_fragmentos(
                        resultado.fragmentos, subdividir_largos=cfg["subdividir"]
                    ),
                }
            )

        resumen = metricas.evaluar(resultados, juicios)
        corridas.append(
            Corrida(
                cfg=cfg,
                ndcg=resumen["ndcg@10"],
                f1=resumen["f1@3"],
                ic_ndcg=resumen["ic_ndcg@10"],
                ic_f1=resumen["ic_f1@3"],
            )
        )

    puntos = puntos_borda(corridas)
    orden = sorted(range(len(corridas)), key=lambda i: puntos[i], reverse=True)

    print(f"\n{'':<3} {'configuración':<52} {'NDCG@10':>8} {'F1@3':>8} {'Borda':>6}")
    for posicion, i in enumerate(orden[: args.top], start=1):
        c = corridas[i]
        print(f"{posicion:>3} {c.etiqueta:<52} {c.ndcg:>8.4f} {c.f1:>8.4f} {puntos[i]:>6}")

    mejor = corridas[orden[0]]
    empatadas = [corridas[i] for i in orden[1:] if indistinguibles(mejor, corridas[i])]

    print(f"\nmejor: {mejor.etiqueta}")
    print(f"  NDCG@10 {mejor.ndcg:.4f}  IC [{mejor.ic_ndcg[0]:.4f}, {mejor.ic_ndcg[1]:.4f}]")
    print(f"  F1@3    {mejor.f1:.4f}  IC [{mejor.ic_f1[0]:.4f}, {mejor.ic_f1[1]:.4f}]")

    if empatadas:
        print(
            f"\n  {len(empatadas)} configuraciones son indistinguibles de esta: sus"
        )
        print("  intervalos se solapan en ambas métricas. Con este número de")
        print("  consultas no hay evidencia para preferir una sobre otra, así que")
        print("  conviene quedarse con la más simple de entre las empatadas.")

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as fh:
        for i in orden:
            c = corridas[i]
            fh.write(
                json.dumps(
                    {
                        **c.cfg,
                        "ndcg@10": c.ndcg,
                        "f1@3": c.f1,
                        "ic_ndcg@10": list(c.ic_ndcg),
                        "ic_f1@3": list(c.ic_f1),
                        "borda": puntos[i],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"\n-> {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
