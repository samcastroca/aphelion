"""Genera el pool de candidatos que el equipo debe anotar.

El ground truth oficial no es público. Para poder medir, se anota a mano sobre las
50 consultas reales.

El pool se construye uniendo el top-N de **cada encoder por separado**, no del
sistema fusionado. Esto mitiga el sesgo de evaluar únicamente lo que la
configuración actual ya encuentra: si solo se anotara la salida final, cualquier
cambio que recuperase documentos nuevos aparecería como ruido no anotado y se
penalizaría injustamente.

Produce un CSV por anotador, repartido de forma que cada consulta tenga un
responsable y una fracción quede duplicada para medir acuerdo entre anotadores.

Uso:
    uv run python scripts/05_pool_anotacion.py --anotadores 4 --top 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from aphelion import config, consultas as mod_consultas, encoders, recuperacion, vectores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="candidatos por encoder")
    ap.add_argument("--anotadores", type=int, default=4)
    ap.add_argument("--solape", type=float, default=0.2, help="fracción duplicada")
    ap.add_argument("--salida", type=Path, default=config.TRABAJO / "anotacion")
    args = ap.parse_args()

    disponibles = vectores.encoders_disponibles()
    if not disponibles:
        print("no hay índices construidos; ejecuta antes scripts/03_indexar.py")
        return 1

    indices = recuperacion.cargar_indices(disponibles)
    preguntas = mod_consultas.cargar()
    args.salida.mkdir(parents=True, exist_ok=True)

    filas: list[dict] = []
    for pregunta in preguntas:
        vistos: set[str] = set()
        for nombre, indice in indices.items():
            vector = encoders.cargar(nombre).codificar_consultas([pregunta.texto])[0]
            for posicion, (idx, similitud) in enumerate(
                indice.buscar(vector, args.top), start=1
            ):
                meta = indice.meta(idx)
                if meta["chunk_id"] in vistos:
                    continue
                vistos.add(meta["chunk_id"])
                filas.append(
                    {
                        "query_id": pregunta.query_id,
                        "pregunta": pregunta.texto,
                        "encoder": nombre,
                        "rango": posicion,
                        "similitud": round(similitud, 4),
                        "doc_id": meta["doc_id"],
                        "fuente": meta["fuente"],
                        "chunk_id": meta["chunk_id"],
                        "idioma": meta.get("idioma", ""),
                        "texto": meta["texto"][:1200],
                        # columnas a rellenar por el anotador
                        "relevancia": "",  # 0 = no, 1 = parcial, 2 = relevante
                        "notas": "",
                    }
                )

    if not filas:
        print("el pool quedó vacío")
        return 1

    # Reparto por consulta, con una fracción duplicada para medir acuerdo.
    orden = [p.query_id for p in preguntas]
    asignacion = {query_id: [i % args.anotadores] for i, query_id in enumerate(orden)}

    for query_id in orden[: int(len(orden) * args.solape)]:
        segundo = (asignacion[query_id][0] + 1) % args.anotadores
        if segundo not in asignacion[query_id]:
            asignacion[query_id].append(segundo)

    campos = list(filas[0].keys())
    for anotador in range(args.anotadores):
        suyas = {q for q, duenos in asignacion.items() if anotador in duenos}
        propias = [f for f in filas if f["query_id"] in suyas]
        destino = args.salida / f"anotador_{anotador + 1}.csv"
        with destino.open("w", encoding="utf-8-sig", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(propias)
        print(
            f"anotador {anotador + 1}: {len(suyas):>2} consultas, "
            f"{len(propias):>4} filas -> {destino}"
        )

    print(f"\ntotal de juicios a emitir: {len(filas):,}")
    print("Rellena la columna 'relevancia': 0 = no relevante, 1 = parcial, 2 = relevante.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
