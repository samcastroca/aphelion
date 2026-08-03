"""Limpia y fragmenta los documentos extraídos.

Produce `trabajo/fragmentos.jsonl`, una línea por fragmento con toda la metadata
obligatoria de la Tabla 1 más las extensiones propias.

El trabajo se reparte entre procesos porque la tokenización es intensiva en CPU y
esta etapa se repite en cada barrido de tamaño de chunk. Cada worker carga su
propio tokenizador una sola vez.

Uso:
    uv run python scripts/etapas/03_fragmentar.py
    uv run python scripts/etapas/03_fragmentar.py --max-tokens 768 --salida trabajo/frag_768.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from aphelion import config
from aphelion.indice import chunking
from aphelion.ingesta import limpieza

# Estado por proceso: evita reenviar los parámetros en cada tarea.
_OPCIONES: dict = {}


def _inicializar(encoder: str, max_tokens: int) -> None:
    _OPCIONES["encoder"] = encoder
    _OPCIONES["max_tokens"] = max_tokens
    # Fuerza la carga del tokenizador aquí y no en la primera tarea, para que el
    # coste aparezca una vez por worker y no distorsione la barra de progreso.
    chunking._tokenizador(encoder)


def _procesar(ruta: str) -> tuple[str, list[dict]]:
    """Devuelve (estado, fragmentos) para un documento."""
    doc = json.loads(Path(ruta).read_text(encoding="utf-8"))

    if doc.get("error"):
        return "error_extraccion", []
    if doc.get("necesita_ocr"):
        return "pendiente_ocr", []
    if not (doc.get("texto") or "").strip():
        return "vacio", []

    texto, idioma = limpieza.limpiar(doc["texto"])
    if not texto.strip():
        return "vacio_tras_limpieza", []

    fragmentos = chunking.fragmentar(
        doc,
        texto,
        idioma,
        nombre_encoder=_OPCIONES["encoder"],
        max_tokens=_OPCIONES["max_tokens"],
    )
    if not fragmentos:
        return "sin_fragmentos", []

    return "ok", [f.to_dict() for f in fragmentos]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=config.CHUNK_PRESUPUESTO)
    ap.add_argument("--salida", type=Path, default=config.FRAGMENTOS)
    ap.add_argument("--encoder", default=config.ENCODER_PRINCIPAL)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    archivos = sorted(config.TEXTO_CRUDO.glob("*.json"))
    if not archivos:
        print("No hay texto extraído. Ejecuta antes scripts/etapas/01_extraer.py")
        return 1

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    print(f"documentos: {len(archivos):,} | procesos: {args.procesos} | "
          f"max_tokens: {args.max_tokens}")

    # El tokenizador se carga aquí antes de abrir el pool. Si no, los workers
    # salen a por él a la vez y en una máquina nueva son N descargas simultáneas
    # del mismo archivo; con la caché caliente, N comprobaciones al Hub que
    # dejan la barra parada en 0% y sueltan un aviso por proceso.
    print("tokenizador ...", end="", flush=True)
    chunking._tokenizador(args.encoder)
    print(" listo")

    estados: Counter[str] = Counter()
    idiomas: Counter[str] = Counter()
    por_formato: Counter[str] = Counter()
    tokens: list[int] = []
    n_frag = 0

    with args.salida.open("w", encoding="utf-8") as fh:
        with ProcessPoolExecutor(
            max_workers=args.procesos,
            initializer=_inicializar,
            initargs=(args.encoder, args.max_tokens),
        ) as pool:
            tareas = pool.map(_procesar, [str(a) for a in archivos], chunksize=4)
            for estado, fragmentos in tqdm(
                tareas, total=len(archivos), desc="fragmentando", unit="doc"
            ):
                estados[estado] += 1
                if not fragmentos:
                    continue
                idiomas[fragmentos[0]["idioma"]] += 1
                por_formato[fragmentos[0]["formato"]] += len(fragmentos)
                for fragmento in fragmentos:
                    fh.write(json.dumps(fragmento, ensure_ascii=False) + "\n")
                    tokens.append(fragmento["num_tokens"])
                    n_frag += 1

    print("\n--- documentos ---")
    for estado, n in estados.most_common():
        print(f"  {estado:<22} {n:>6}")

    print("\n--- fragmentos ---")
    print(f"  total                  {n_frag:>7,}")
    if tokens:
        tokens.sort()
        print(f"  tokens min/mediana/max {tokens[0]} / {tokens[len(tokens) // 2]} / {tokens[-1]}")
        print(f"  tokens totales         {sum(tokens):>7,}")
    print(f"  idiomas                {dict(idiomas)}")
    print(f"  por formato            {dict(por_formato)}")
    print(f"\n-> {args.salida}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
