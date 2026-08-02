"""Codifica los fragmentos y construye el índice FAISS.

Los embeddings se calculan por lotes y se cachean en disco, de modo que una
interrupción no obliga a recodificar el corpus completo. Esto importa porque la
codificación es la etapa cara y se ejecuta en una máquina distinta a la de
desarrollo.

Uso:
    uv run python scripts/03_indexar.py [--encoder bge-m3] [--lote 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aphelion import config, encoders, vectores


def leer_fragmentos(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def codificar_por_lotes(
    encoder: encoders.Encoder,
    textos: list[str],
    cache: Path,
    tam_lote: int,
    frag_por_bloque: int = 2048,
) -> np.ndarray:
    """Codifica en bloques persistidos, reanudable ante interrupciones."""
    cache.mkdir(parents=True, exist_ok=True)
    bloques: list[np.ndarray] = []

    total_bloques = (len(textos) + frag_por_bloque - 1) // frag_por_bloque
    for n in range(total_bloques):
        destino = cache / f"bloque_{n:05d}.npy"
        if destino.exists():
            bloques.append(np.load(destino))
            continue

        inicio = n * frag_por_bloque
        trozo = textos[inicio : inicio + frag_por_bloque]

        t0 = time.time()
        bloque = encoder.codificar_pasajes(trozo, tam_lote=tam_lote, progreso=False)
        np.save(destino, bloque)
        bloques.append(bloque)

        hechos = min(inicio + frag_por_bloque, len(textos))
        velocidad = len(trozo) / max(time.time() - t0, 1e-6)
        restante = (len(textos) - hechos) / max(velocidad, 1e-6)
        print(
            f"  bloque {n + 1}/{total_bloques}  {hechos:,}/{len(textos):,} fragmentos"
            f"  ({velocidad:.0f} frag/s, faltan ~{restante / 60:.1f} min)",
            flush=True,
        )

    return np.vstack(bloques)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=config.ENCODER_PRINCIPAL)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--fragmentos", type=Path, default=config.FRAGMENTOS)
    args = ap.parse_args()

    if not args.fragmentos.exists():
        print(f"No existe {args.fragmentos}. Ejecuta antes scripts/02_fragmentar.py")
        return 1

    fragmentos = leer_fragmentos(args.fragmentos)
    print(f"fragmentos: {len(fragmentos):,}")

    encoder = encoders.cargar(args.encoder)
    print(f"encoder:    {encoder.cfg['modelo']}  ({encoder.dim}d)")
    print(f"dispositivo: {encoder.device}")
    if encoder.device == "cpu":
        print("  aviso: sin GPU la codificación es varias veces más lenta")

    textos = [f["texto"] for f in fragmentos]
    cache = config.TRABAJO / "embeddings" / args.encoder

    t0 = time.time()
    matriz = codificar_por_lotes(encoder, textos, cache, args.lote)
    print(f"codificado en {(time.time() - t0) / 60:.1f} min")

    # Verificación: los vectores deben tener norma unitaria para que el producto
    # interno del índice sea la similitud coseno.
    normas = np.linalg.norm(matriz, axis=1)
    if not np.allclose(normas, 1.0, atol=1e-3):
        print(f"  aviso: normas fuera de rango [{normas.min():.4f}, {normas.max():.4f}]")

    index = vectores.construir(matriz, encoder.dim)
    destino = vectores.guardar(args.encoder, index, fragmentos)

    print(f"\nvectores indexados: {index.ntotal:,}")
    print(f"-> {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
