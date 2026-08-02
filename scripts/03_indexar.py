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
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from aphelion import config, encoders, vectores


def leer_fragmentos(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def huella(ruta: Path) -> str:
    """Identifica el archivo de fragmentos que originó una caché de embeddings.

    Sin esto la caché se indexa solo por número de bloque, y una corrida sobre la
    muestra deja `bloque_00000.npy` con 1168 vectores que la corrida siguiente
    sobre el corpus completo reutiliza como si fueran los primeros 2048. La
    desalineación se detecta al guardar, pero después de recodificar el corpus
    entero: horas de GPU perdidas por un nombre de archivo reutilizado.
    """
    h = hashlib.blake2b(digest_size=8)
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


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
        inicio = n * frag_por_bloque
        trozo = textos[inicio : inicio + frag_por_bloque]

        destino = cache / f"bloque_{n:05d}.npy"
        if destino.exists():
            cacheado = np.load(destino)
            # Cinturón además de los tirantes de `huella()`: un bloque cacheado
            # con otro número de filas desalinearía índice y metadata en silencio.
            if len(cacheado) != len(trozo):
                raise ValueError(
                    f"{destino} tiene {len(cacheado)} vectores y se esperaban "
                    f"{len(trozo)}. Borra la carpeta {cache} y recodifica."
                )
            bloques.append(cacheado)
            continue

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
    ap.add_argument(
        "--backend",
        choices=("torch", "onnx"),
        default="torch",
        help="onnx usa la GPU Radeon vía DirectML (uv sync --extra amd)",
    )
    args = ap.parse_args()

    if not args.fragmentos.exists():
        print(f"No existe {args.fragmentos}. Ejecuta antes scripts/02_fragmentar.py")
        return 1

    fragmentos = leer_fragmentos(args.fragmentos)
    print(f"fragmentos: {len(fragmentos):,}")
    textos = [f["texto"] for f in fragmentos]

    if args.backend == "onnx":
        from aphelion import onnx_dml

        # La divergencia entre backends no rompe nada visiblemente: deja el
        # índice y las consultas del jurado en espacios distintos. Se comprueba
        # antes de codificar, no después.
        #
        # La referencia de PyTorch se calcula y se libera *antes* de abrir la
        # sesión ONNX: las dos copias del modelo juntas no caben en 16 GB.
        muestra = textos[:8]
        print("referencia con PyTorch ...", flush=True)
        referencia = onnx_dml.referencia_torch(args.encoder, muestra)

        print("exportando/abriendo el modelo ONNX ...", flush=True)
        encoder = onnx_dml.EncoderONNX(args.encoder)
        ok, peor = onnx_dml.verificar(referencia, encoder, muestra)
        print(f"fidelidad frente a PyTorch: coseno mínimo {peor:.6f}")
        if not ok:
            print("  los dos backends divergen; se aborta antes de construir el índice")
            return 1
    else:
        encoder = encoders.cargar(args.encoder)

    print(f"encoder:    {encoder.cfg['modelo']}  ({encoder.dim}d)")
    print(f"dispositivo: {encoder.device}")
    if encoder.device == "cpu":
        print("  aviso: sin GPU la codificación es varias veces más lenta")

    # La caché depende del backend: mezclar bloques de PyTorch y de DirectML
    # dejaría el índice con vectores de dos procedencias distintas.
    cache = (
        config.TRABAJO
        / "embeddings"
        / f"{args.encoder}-{args.backend}"
        / huella(args.fragmentos)
    )
    print(f"caché:      {cache}")

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
