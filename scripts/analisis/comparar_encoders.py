"""Genera un archivo de resultados por cada encoder por separado, y el fusionado.

Sirve para dos cosas distintas:

1. **Medir qué aporta la fusión.** Con `resultados_bge-m3.jsonl`,
   `resultados_me5-large.jsonl` y `resultados_fusion.jsonl` evaluados contra el
   mismo ground truth, la pregunta «¿el segundo encoder paga su coste?» deja de
   responderse por argumento y pasa a responderse por número. Si la fusión no
   supera al mejor de los dos por separado, sobra medio índice y la mitad del
   tiempo de codificación.

2. **Alimentar el pool de anotación.** La búsqueda es la parte cara —cargar dos
   modelos de 2,2 GB y codificar 50 consultas—, así que se hace una sola vez y
   los rankings crudos se cachean en `trabajo/rankings.jsonl`. De ahí sale el
   pool de `pool_juicios.py` sin volver a tocar la GPU ni los modelos.

La política de recuperación es la misma en los tres casos: la del paquete
`aphelion`. Con un solo índice, RRF sobre un único ranking es una transformación
monótona del rango, así que el orden lo decide ese encoder; el resto de la
política —realce por fenómeno, deduplicación, diversificación, max pooling— se
aplica igual. Eso es lo que hace justa la comparación: lo único que cambia entre
las tres corridas es de dónde salen los candidatos.

Uso:
    uv run python scripts/analisis/comparar_encoders.py
    uv run python scripts/analisis/comparar_encoders.py --profundidad 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas, recuperacion, salida

CACHE_RANKINGS = config.TRABAJO / "rankings.jsonl"


def buscar_todo(
    recuperador: recuperacion.Recuperador,
    preguntas: list,
    profundidad: int,
) -> dict[str, dict[str, list[tuple[dict, float]]]]:
    """Una búsqueda por consulta y encoder. Es lo único caro de este script."""
    cache: dict[str, dict[str, list[tuple[dict, float]]]] = {}
    for pregunta in tqdm(preguntas, desc="buscando", unit="q"):
        cache[pregunta.query_id] = recuperador.buscar(
            pregunta, candidatos_por_indice=profundidad
        )
    return cache


def guardar_rankings(
    cache: dict[str, dict[str, list[tuple[dict, float]]]], destino: Path
) -> Path:
    """Persiste los rankings crudos por referencia, no por contenido.

    Se guardan `chunk_id` y similitud, no el texto: el texto ya vive en
    `metadata.jsonl` y duplicarlo aquí costaría cientos de megabytes para no
    añadir nada. Quien lea este archivo resuelve el texto por `chunk_id`.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for query_id, rankings in cache.items():
            fh.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "rankings": {
                            encoder: [
                                {"chunk_id": meta["chunk_id"], "sim": round(sim, 6)}
                                for meta, sim in lista
                            ]
                            for encoder, lista in rankings.items()
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return destino


def generar(
    recuperador: recuperacion.Recuperador,
    preguntas: list,
    cache: dict,
    encoders: list[str] | None,
) -> list[dict]:
    """Ordena el pool cacheado usando solo los encoders pedidos."""
    registros = []
    for pregunta in preguntas:
        rankings = cache[pregunta.query_id]
        if encoders is not None:
            rankings = {k: v for k, v in rankings.items() if k in encoders}
        resultado = recuperador.ordenar(rankings, pregunta)
        registros.append(salida.resultado_a_dict(resultado))
    return registros


def solape(a: list[dict], b: list[dict], campo: str, clave: str) -> float:
    """Fracción media de elementos compartidos entre dos corridas, por consulta."""
    por_query_b = {r["query_id"]: r for r in b}
    fracciones = []
    for registro in a:
        otro = por_query_b.get(registro["query_id"])
        if otro is None:
            continue
        uno = {x[clave] for x in registro[campo]}
        dos = {x[clave] for x in otro[campo]}
        if not uno:
            continue
        fracciones.append(len(uno & dos) / len(uno))
    return sum(fracciones) / len(fracciones) if fracciones else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--consultas", type=Path, default=config.PREGUNTAS_PDF)
    ap.add_argument("--salida", type=Path, default=config.TRABAJO)
    ap.add_argument(
        "--encoders",
        help="lista separada por comas; por defecto todos los índices encontrados",
    )
    ap.add_argument(
        "--profundidad",
        type=int,
        default=config.CANDIDATOS_POR_INDICE,
        help="candidatos por índice; también fija la profundidad del pool cacheado",
    )
    args = ap.parse_args()

    elegidos = [e.strip() for e in args.encoders.split(",")] if args.encoders else None
    indices = recuperacion.cargar_indices(elegidos, args.base)
    nombres = sorted(indices)
    print(f"índices: {', '.join(f'{n} ({len(i):,})' for n, i in indices.items())}")

    preguntas = mod_consultas.cargar(args.consultas)
    print(f"consultas: {len(preguntas)}")

    recuperador = recuperacion.Recuperador(indices)

    t0 = time.time()
    cache = buscar_todo(recuperador, preguntas, args.profundidad)
    print(f"búsqueda completa en {time.time() - t0:.1f}s")

    ruta_cache = guardar_rankings(cache, CACHE_RANKINGS)
    print(f"rankings crudos -> {ruta_cache}")

    args.salida.mkdir(parents=True, exist_ok=True)
    corridas: dict[str, list[dict]] = {}

    for nombre in nombres:
        registros = generar(recuperador, preguntas, cache, [nombre])
        destino = args.salida / f"resultados_{nombre}.jsonl"
        salida.escribir_jsonl(registros, destino)
        corridas[nombre] = registros
        print(f"  {nombre:<12} -> {destino}")

    # Con un solo índice la fusión no fusiona nada: sería una copia del anterior.
    if len(nombres) > 1:
        registros = generar(recuperador, preguntas, cache, None)
        destino = args.salida / "resultados_fusion.jsonl"
        salida.escribir_jsonl(registros, destino)
        corridas["fusion"] = registros
        print(f"  {'fusion':<12} -> {destino}")

    for nombre, registros in corridas.items():
        problemas = salida.validar(
            args.salida / f"resultados_{nombre}.jsonl", len(preguntas)
        )
        if problemas:
            print(f"\n{nombre}: {len(problemas)} problemas de formato")
            for problema in problemas[:5]:
                print(f"  - {problema}")

    # Cuánto se parecen entre sí. Un solape alto entre los dos encoders
    # significaría que el segundo aporta poco y que su coste no se justifica.
    print(f"\n{'':<26} {'frag@10':>8} {'docs@3':>8}")
    etiquetas = list(corridas)
    for i, a in enumerate(etiquetas):
        for b in etiquetas[i + 1 :]:
            frag = solape(corridas[a], corridas[b], "fragments", "chunk_id")
            docs = solape(corridas[a], corridas[b], "documents", "doc_id")
            print(f"  {a} vs {b:<15} {frag:>8.1%} {docs:>8.1%}")

    print(f"\nSiguiente paso: uv run python scripts/analisis/pool_juicios.py --generar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
