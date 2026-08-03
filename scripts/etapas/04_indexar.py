"""Codifica los fragmentos y construye el índice FAISS.

Los embeddings se calculan por lotes y se cachean en disco, de modo que una
interrupción no obliga a recodificar el corpus completo. Esto importa porque la
codificación es la etapa cara y se ejecuta en una máquina distinta a la de
desarrollo.

Esa misma caché por bloques permite repartir la codificación entre varias
máquinas: `--reparto 0:40` codifica el primer 40% de los bloques y no construye
el índice. Cada máquina se queda con un tramo, los `.npy` se juntan en una sola
carpeta y una corrida sin `--reparto` los encuentra cacheados y arma el índice
en segundos. Los tramos no tienen que ser iguales: quien tenga mejor GPU carga
con más porcentaje.

La condición que no se puede saltar es que las tres máquinas partan del **mismo**
`fragmentos.jsonl`, distribuido como archivo y no regenerado en cada una. La
huella que da nombre a la carpeta de caché lo delata —dos archivos distintos dan
dos carpetas distintas— pero conviene compararla antes de gastar la GPU.

Uso:
    uv run python scripts/etapas/04_indexar.py [--encoder bge-m3] [--lote 32]
    uv run python scripts/etapas/04_indexar.py --reparto 0:40    # solo su tramo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from aphelion import config
from aphelion.indice import encoders, vectores


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


FRAG_POR_BLOQUE = 2048


def numero_de_bloques(n_fragmentos: int) -> int:
    return (n_fragmentos + FRAG_POR_BLOQUE - 1) // FRAG_POR_BLOQUE


def porcentajes_de_reparto(reparto: str) -> tuple[float, float]:
    """'0:40' -> (0.0, 40.0). Aborta si no es un tramo utilizable.

    Se comprueba aparte del cálculo de bloques para poder llamarlo nada más
    arrancar: un `0-40` en vez de `0:40` no debe descubrirse después de haber
    cargado el modelo.
    """
    try:
        crudo_inicio, crudo_fin = reparto.split(":")
        p_inicio, p_fin = float(crudo_inicio), float(crudo_fin)
    except ValueError:
        raise SystemExit(f"--reparto mal formado: {reparto!r}. Se espera algo como 0:40")

    if not 0 <= p_inicio < p_fin <= 100:
        raise SystemExit(f"--reparto fuera de rango: {reparto!r}. Se espera 0 <= a < b <= 100")

    return p_inicio, p_fin


def tramo_de_bloques(reparto: str, total_bloques: int) -> range:
    """'0:40' -> los bloques que caen en el primer 40% del corpus.

    Los extremos se redondean hacia abajo sobre el total de bloques, así que
    tramos contiguos —0:40, 40:70, 70:100— cubren todos los bloques exactamente
    una vez sin que las máquinas tengan que ponerse de acuerdo en nada más.
    """
    p_inicio, p_fin = porcentajes_de_reparto(reparto)
    return range(
        int(total_bloques * p_inicio / 100),
        int(total_bloques * p_fin / 100),
    )


def codificar_bloques(
    encoder: encoders.Encoder,
    textos: list[str],
    cache: Path,
    tam_lote: int,
    bloques: range,
) -> list[np.ndarray]:
    """Codifica los bloques pedidos, saltando los que ya estén en la caché."""
    cache.mkdir(parents=True, exist_ok=True)
    total_bloques = numero_de_bloques(len(textos))
    codificados: list[np.ndarray] = []
    pendientes = [n for n in bloques if not (cache / f"bloque_{n:05d}.npy").exists()]
    hechos_ahora = 0

    for n in bloques:
        inicio = n * FRAG_POR_BLOQUE
        trozo = textos[inicio : inicio + FRAG_POR_BLOQUE]

        destino = cache / f"bloque_{n:05d}.npy"
        if destino.exists():
            cacheado = np.load(destino)
            # Cinturón además de los tirantes de `huella()`: un bloque cacheado
            # con otro número de filas desalinearía índice y metadata en silencio.
            if len(cacheado) != len(trozo):
                raise ValueError(
                    f"{destino} tiene {len(cacheado)} vectores y se esperaban "
                    f"{len(trozo)}. Borra ese archivo y recodifícalo."
                )
            codificados.append(cacheado)
            continue

        t0 = time.time()
        bloque = encoder.codificar_pasajes(trozo, tam_lote=tam_lote, progreso=False)
        np.save(destino, bloque)
        codificados.append(bloque)
        hechos_ahora += 1

        velocidad = len(trozo) / max(time.time() - t0, 1e-6)
        faltan = sum(len(textos[m * FRAG_POR_BLOQUE :][:FRAG_POR_BLOQUE]) for m in pendientes[hechos_ahora:])
        print(
            f"  bloque {n + 1}/{total_bloques}  {hechos_ahora}/{len(pendientes)} de este tramo"
            f"  ({velocidad:.0f} frag/s, faltan ~{faltan / max(velocidad, 1e-6) / 60:.1f} min)",
            flush=True,
        )

    return codificados


def bloques_faltantes(cache: Path, total_bloques: int) -> list[int]:
    return [n for n in range(total_bloques) if not (cache / f"bloque_{n:05d}.npy").exists()]


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
    ap.add_argument(
        "--reparto",
        metavar="A:B",
        help="codifica solo el tramo A%%-B%% de los bloques y no construye el índice",
    )
    ap.add_argument(
        "--huella",
        action="store_true",
        help="imprime la huella de los fragmentos y sale, para comparar entre máquinas",
    )
    args = ap.parse_args()

    # Antes que nada y sin cargar nada: comparar esta línea entre las máquinas
    # que se reparten el trabajo cuesta un segundo y decide si sus vectores van a
    # encajar. Si no coincide, sobra seguir.
    if args.huella:
        if not args.fragmentos.exists():
            print(f"No existe {args.fragmentos}")
            return 1
        fragmentos = leer_fragmentos(args.fragmentos)
        print(f"huella:   {huella(args.fragmentos)}")
        print(f"fragmentos: {len(fragmentos):,}")
        print(f"bloques:  {numero_de_bloques(len(fragmentos))} de {FRAG_POR_BLOQUE}")
        return 0

    # Lo primero, antes de leer fragmentos y de cargar el modelo: un tramo mal
    # escrito no puede costar los minutos que tarda el encoder en abrirse.
    if args.reparto:
        porcentajes_de_reparto(args.reparto)

    if not args.fragmentos.exists():
        print(f"No existe {args.fragmentos}. Ejecuta antes scripts/etapas/03_fragmentar.py")
        return 1

    fragmentos = leer_fragmentos(args.fragmentos)
    print(f"fragmentos: {len(fragmentos):,}")
    textos = [f["texto"] for f in fragmentos]

    if args.backend == "onnx":
        from aphelion.indice import onnx_dml

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
    hue = huella(args.fragmentos)
    cache = config.TRABAJO / "embeddings" / f"{args.encoder}-{args.backend}" / hue
    total_bloques = numero_de_bloques(len(fragmentos))
    print(f"huella:     {hue}  (tiene que coincidir en todas las máquinas)")
    print(f"caché:      {cache}")
    print(f"bloques:    {total_bloques} de {FRAG_POR_BLOQUE} fragmentos")

    if args.reparto:
        tramo = tramo_de_bloques(args.reparto, total_bloques)
        if not tramo:
            print(f"  el tramo {args.reparto} no cubre ningún bloque de los {total_bloques}")
            return 1
        print(f"reparto:    {args.reparto}%  ->  bloques {tramo.start}..{tramo.stop - 1}")

        t0 = time.time()
        codificar_bloques(encoder, textos, cache, args.lote, tramo)
        print(f"\ntramo codificado en {(time.time() - t0) / 60:.1f} min")
        print(f"-> copia los .npy de {cache} a la máquina que arma el índice")
        return 0

    faltan = bloques_faltantes(cache, total_bloques)
    print(f"en caché:   {total_bloques - len(faltan)}/{total_bloques} bloques")
    if faltan:
        # Al fusionar tramos de varias máquinas, un hueco aquí significa que un
        # tramo no llegó. Se codifica igual —es lo correcto— pero conviene verlo
        # antes de que la GPU se pase una hora rellenando lo que ya existía.
        print(f"  faltan {len(faltan)}: {faltan[:10]}{' ...' if len(faltan) > 10 else ''}")

    t0 = time.time()
    matriz = np.vstack(
        codificar_bloques(encoder, textos, cache, args.lote, range(total_bloques))
    )
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
