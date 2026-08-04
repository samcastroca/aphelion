"""Comprueba que `generador.py` y el paquete `aphelion` producen lo mismo.

`generador.py` es autónomo a propósito —el jurado recibe solo `entrega/`— y eso
significa que la política de recuperación está escrita dos veces: una en
`aphelion.recuperacion`/`aphelion.salida`, que es la que se itera durante el
desarrollo, y otra dentro del entregable. Dos copias divergen solas.

Este script corre ambas sobre el mismo índice y exige salidas idénticas. Si
falla, el entregable ya no refleja el sistema que se afinó.

Corre como última etapa del pipeline, sobre el índice que acaba de quedar en la
entrega.

Uso:
    uv run python scripts/etapas/06_verificar.py
    uv run python scripts/etapas/06_verificar.py --base otra/base_vectorial
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas, recuperacion, salida


def cargar_generador():
    """Importa generador.py por ruta: no es un módulo del paquete a propósito."""
    ruta = config.ENTREGA / "generador.py"
    nombre = "generador_entregable"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    # Registrar antes de ejecutar: `dataclasses` resuelve las anotaciones vía
    # `sys.modules[cls.__module__]` y falla si el módulo todavía no está ahí.
    sys.modules[nombre] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def salida_del_paquete(base: Path, preguntas: Path) -> list[dict]:
    """Corre la política del paquete y **libera los modelos antes de volver**.

    Las dos implementaciones cargan su propia copia de cada encoder, así que
    ejecutarlas seguidas mantiene cuatro modelos de ~2,2 GB residentes a la vez.
    En una GPU de 4 GB eso muere con un `CUDA out of memory` en la segunda mitad
    —después de haber codificado las 50 consultas de la primera—, y el fallo no
    tiene nada que ver con lo que este script comprueba. Se sueltan aquí.
    """
    import gc

    from aphelion.indice import encoders

    indices = recuperacion.cargar_indices(None, base)
    recuperador = recuperacion.Recuperador(indices)
    registros = [
        salida.resultado_a_dict(recuperador.recuperar(c))
        for c in mod_consultas.cargar(preguntas)
    ]

    recuperador._encoders.clear()
    encoders.cargar.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return registros


def salida_del_entregable(base: Path, preguntas: Path) -> list[dict]:
    gen = cargar_generador()
    gen.set_seeds()
    indices = {n: gen.load_index(n, base) for n in gen.available_encoders(base)}
    recuperador = gen.Retriever(indices)
    registros = []
    for consulta in gen.load_queries(preguntas):
        documentos, fragmentos = recuperador.retrieve(consulta)
        registros.append(gen.build_record(consulta.query_id, documentos, fragmentos))
    return registros


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--consultas", type=Path, default=config.PREGUNTAS_PDF)
    args = ap.parse_args()

    if not args.base.exists():
        print(f"no existe la base vectorial: {args.base}")
        return 1

    print(f"base: {args.base}")
    print("paquete    ...", flush=True)
    a = salida_del_paquete(args.base, args.consultas)
    print("entregable ...", flush=True)
    b = salida_del_entregable(args.base, args.consultas)

    if len(a) != len(b):
        print(f"\nDIVERGEN: {len(a)} consultas frente a {len(b)}")
        return 1

    diferencias = [x["query_id"] for x, y in zip(a, b) if x != y]
    if diferencias:
        print(f"\nDIVERGEN en {len(diferencias)} consultas: {diferencias[:10]}")
        volcado = Path(tempfile.mkdtemp()) / "divergencia.json"
        primera = next(x["query_id"] for x, y in zip(a, b) if x != y)
        volcado.write_text(
            json.dumps(
                {
                    "query_id": primera,
                    "paquete": next(x for x in a if x["query_id"] == primera),
                    "entregable": next(y for y in b if y["query_id"] == primera),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"primera divergencia volcada en {volcado}")
        return 1

    print(f"\nIDÉNTICO en las {len(a)} consultas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
