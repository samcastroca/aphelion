#!/usr/bin/env python
"""Genera `resultados.jsonl` a partir de la base vectorial persistida.

Entregable exigido por la §1.4. Carga los índices FAISS ya construidos, lee el
archivo de consultas y produce el archivo de resultados sin reindexar el corpus.
La especificación excluye de la evaluación las entregas que no reproducen sus
resultados, de modo que este script es la garantía de reproducibilidad.

Uso:
    python generador.py
    python generador.py --consultas ruta/consultas.pdf --salida entrega/resultados.jsonl
    python generador.py --encoders bge-m3            # un solo índice
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from aphelion import config, consultas as mod_consultas, recuperacion, salida  # noqa: E402

SEMILLA = 20260801


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """La recuperación es determinista, pero las semillas se fijan igualmente
    para que cualquier componente futuro con aleatoriedad no rompa la
    reproducibilidad exigida."""
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch

        torch.manual_seed(semilla)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(semilla)
    except ImportError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--consultas", type=Path, default=config.PREGUNTAS_PDF)
    ap.add_argument("--salida", type=Path, default=config.RESULTADOS)
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--encoders", help="lista separada por comas; por defecto todos")
    ap.add_argument("--k0", type=int, default=config.RRF_K0)
    ap.add_argument("--boost", type=float, default=config.BOOST_FENOMENO)
    ap.add_argument("--max-por-doc", type=int, default=config.MAX_FRAGMENTOS_POR_DOC)
    args = ap.parse_args()

    fijar_semillas()

    preguntas = mod_consultas.cargar(args.consultas)
    print(f"consultas: {len(preguntas)}")

    nombres = [e.strip() for e in args.encoders.split(",")] if args.encoders else None
    indices = recuperacion.cargar_indices(nombres, args.base)
    for nombre, indice in indices.items():
        print(f"índice {nombre}: {len(indice):,} fragmentos")

    recuperador = recuperacion.Recuperador(
        indices, k0=args.k0, boost=args.boost, max_por_doc=args.max_por_doc
    )

    resultados = []
    for pregunta in preguntas:
        resultado = recuperador.recuperar(pregunta)
        resultados.append(salida.resultado_a_dict(resultado))

    destino = salida.escribir_jsonl(resultados, args.salida)
    print(f"\n-> {destino}")

    problemas = salida.validar(destino, n_consultas=len(preguntas))
    if problemas:
        print(f"\n{len(problemas)} problemas de formato:")
        for problema in problemas[:20]:
            print(f"  - {problema}")
        return 1

    print("formato validado: esquema correcto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
