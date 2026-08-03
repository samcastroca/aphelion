"""Extrae el texto de los 1826 documentos del catálogo.

Guarda un JSON por documento en `trabajo/texto/`, de modo que la limpieza y el
chunking puedan iterarse sin repetir la extracción, que es la etapa cara.

Uso:
    uv run python scripts/etapas/01_extraer.py [--forzar] [--solo pdf,json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from tqdm import tqdm

from aphelion import config
from aphelion.ingesta import catalogo, extraccion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forzar", action="store_true", help="reprocesa lo ya extraído")
    ap.add_argument("--solo", help="lista de formatos separados por coma")
    args = ap.parse_args()

    config.TEXTO_CRUDO.mkdir(parents=True, exist_ok=True)

    docs = catalogo.cargar()
    if args.solo:
        formatos = {f.strip().lower() for f in args.solo.split(",")}
        docs = [d for d in docs if d.formato in formatos]

    estados: Counter[str] = Counter()
    errores: list[tuple[str, str]] = []
    total_chars = 0

    for doc in tqdm(docs, desc="extrayendo", unit="doc"):
        destino = config.TEXTO_CRUDO / f"{doc.doc_id}.json"
        if destino.exists() and not args.forzar:
            estados["cacheado"] += 1
            continue

        res = extraccion.extraer(doc)

        if res.error:
            estados["error"] += 1
            errores.append((doc.doc_id, res.error))
        elif res.necesita_ocr:
            estados["necesita_ocr"] += 1
        elif res.vacia:
            estados["vacio"] += 1
        else:
            estados["ok"] += 1
            total_chars += len(res.texto)

        destino.write_text(
            json.dumps(
                {
                    **doc.to_dict(),
                    "texto": res.texto,
                    "meta": res.meta,
                    "necesita_ocr": res.necesita_ocr,
                    "error": res.error,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print("\n--- resumen ---")
    for estado, n in estados.most_common():
        print(f"  {estado:<14} {n:>5}")
    print(f"  {'chars':<14} {total_chars:>5,}")

    # Qué documentos necesitan OCR queda anotado en su propio JSON, y la etapa
    # siguiente lo lee de ahí. No se escribe ninguna lista aparte: sería un hecho
    # sobre esta corrida y no sobre la caché, y en cuanto la extracción estuviera
    # cacheada —que es lo normal al reanudar— saldría vacía.
    if estados["necesita_ocr"]:
        print(f"\n{estados['necesita_ocr']} documentos requieren OCR")

    if errores:
        print(f"\n{len(errores)} errores:")
        for doc_id, err in errores[:15]:
            print(f"  {doc_id}: {err[:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
