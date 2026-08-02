"""Compara motores de OCR sobre los PDFs escaneados del corpus.

El objetivo no es medir velocidad —son ~1.000 páginas y el proceso corre una sola
vez— sino detectar **alucinación**: un motor que inventa texto introduce evidencia
falsa en el índice, lo que es peor que no extraer nada.

Señales que el informe reporta automáticamente:

- Densidad de diacríticos. Los informes están en español; un motor que devuelve
  texto sin tildes ni eñes está fallando aunque el resultado parezca legible.
- Repetición de n-gramas. Es el modo de fallo característico de los VLM de OCR:
  entran en bucle y repiten la misma frase decenas de veces.
- Proporción de caracteres no imprimibles, que delata ruido de rasterización.

La comparación final es humana: el script vuelca ambas salidas lado a lado para
que se contrasten contra la imagen original.

Uso:
    uv run python scripts/probar_ocr.py --motores tesseract,unlimited --paginas 2
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from aphelion import config, ocr

# Muestra por defecto: uno corto para iterar rápido, uno largo para ver si el
# motor se degrada con documentos extensos.
MUESTRA = [
    "F3_Dinamicas_Territoriales/Alertas_Tempranas/pdfs/Informes/ALERTAS_informes006.pdf",
    "F3_Dinamicas_Territoriales/Alertas_Tempranas/pdfs/Informes/ALERTAS_informes061.pdf",
]

_DIACRITICOS = set("áéíóúñüÁÉÍÓÚÑÜ")


def densidad_diacriticos(texto: str) -> float:
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return 0.0
    return sum(1 for c in letras if c in _DIACRITICOS) / len(letras)


def repeticion_ngramas(texto: str, n: int = 8) -> float:
    """Proporción del texto ocupada por el n-grama más repetido.

    Un valor alto delata un bucle de generación.
    """
    palabras = texto.split()
    if len(palabras) < n * 3:
        return 0.0
    ngramas = Counter(
        " ".join(palabras[i : i + n]) for i in range(len(palabras) - n + 1)
    )
    _, veces = ngramas.most_common(1)[0]
    return veces / max(len(ngramas), 1)


def ratio_no_imprimible(texto: str) -> float:
    if not texto:
        return 0.0
    malos = sum(
        1 for c in texto if unicodedata.category(c) in ("Cc", "Co", "Cn") and c not in "\n\t"
    )
    return malos / len(texto)


def informe(resultado: ocr.ResultadoOCR) -> dict:
    texto = resultado.texto
    palabras = texto.split()
    return {
        "motor": resultado.motor,
        "paginas": resultado.num_paginas,
        "segundos": round(resultado.segundos, 1),
        "seg_por_pagina": round(resultado.segundos / max(resultado.num_paginas, 1), 2),
        "caracteres": len(texto),
        "palabras": len(palabras),
        "diacriticos": round(densidad_diacriticos(texto), 4),
        "repeticion": round(repeticion_ngramas(texto), 4),
        "no_imprimible": round(ratio_no_imprimible(texto), 5),
        "error": resultado.error,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motores", default="tesseract,unlimited")
    ap.add_argument("--paginas", type=int, default=2, help="máximo de páginas por PDF")
    ap.add_argument("--pdfs", help="rutas relativas al corpus, separadas por coma")
    ap.add_argument("--salida", type=Path, default=config.TRABAJO / "ocr_comparacion")
    args = ap.parse_args()

    motores = [m.strip() for m in args.motores.split(",")]
    rutas = [r.strip() for r in args.pdfs.split(",")] if args.pdfs else MUESTRA
    args.salida.mkdir(parents=True, exist_ok=True)

    for rel in rutas:
        ruta = config.CORPUS / rel
        if not ruta.exists():
            print(f"no existe: {rel}")
            continue

        print(f"\n=== {Path(rel).name} ===")
        informes = []

        for motor in motores:
            print(f"  {motor} ...", flush=True)
            resultado = ocr.procesar(ruta, motor=motor, max_paginas=args.paginas)
            datos = informe(resultado)
            informes.append(datos)

            destino = args.salida / f"{Path(rel).stem}__{motor}.txt"
            destino.write_text(resultado.texto, encoding="utf-8")

            if resultado.error:
                print(f"    ERROR: {resultado.error}")
            else:
                print(
                    f"    {datos['palabras']:>6} palabras | "
                    f"{datos['seg_por_pagina']:>5}s/pag | "
                    f"diacriticos {datos['diacriticos']:.3f} | "
                    f"repeticion {datos['repeticion']:.3f}"
                )
                print(f"    -> {destino}")

        validos = [i for i in informes if not i["error"]]
        if len(validos) > 1:
            print("\n  lectura:")
            print("    diacriticos ~0.02-0.06 es normal en espanol; ~0 indica perdida de tildes")
            print("    repeticion  >0.05 sugiere bucle de generacion (alucinacion)")

    print(f"\nSalidas completas en {args.salida}")
    print("Contrastalas contra el PDF original antes de decidir el motor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
