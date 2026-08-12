#!/usr/bin/env python
"""Construye el grafo de conocimiento y lo exporta a GraphML (§7, bonus).

Las tres etapas de la §7.2 en una corrida: NER multilingüe sobre los fragmentos,
extracción de relaciones por patrones y construcción con NetworkX. La salida es
`entrega/base_vectorial/grafo/grafo.graphml`, que es donde la §1.4 la pide.

**No depende del índice.** El grafo se construye sobre los fragmentos, así que
esta etapa está en el mismo escalón que `04_indexar` y puede correr en paralelo
con ella. Por eso comparte número: no es que vaya después, es que va al lado.

**El NER se cachea.** Una pasada sobre los 64.484 fragmentos cuesta lo que una
indexación con encoder base, y reconstruir el grafo con otra poda no debería
pagarla otra vez. Va a `trabajo/entidades.jsonl` y las corridas siguientes la
reutilizan salvo `--rehacer-ner`.

**Backends de NER.** `onnx` usa `fastino/gliner2-multi-v1` exportado (Apache-2.0)
a través de `gliner2-onnx` (MIT), que no arrastra torch y admite el proveedor
DirectML que este proyecto ya monta para la Radeon. Es una dependencia opcional:
si no está instalada, esta etapa lo dice y no rompe el resto del pipeline. El
backend `falso` no descarga nada y sirve para comprobar el esquema y la
exportación de punta a punta antes de gastar GPU.

Uso:
    uv run python scripts/etapas/04_grafo.py                    # NER real, corpus entero
    uv run python scripts/etapas/04_grafo.py --ner falso        # sin modelo, para probar
    uv run python scripts/etapas/04_grafo.py --limite 2000      # una muestra
    uv run python scripts/etapas/04_grafo.py --agrupar-entidades  # fusión cross-lingüe

    # Si se prefiere el backend `gliner` v1, que declara transformers<5.14.0 y
    # por tanto no convive con el entorno del proyecto:
    uv run --isolated --with gliner2-onnx python scripts/etapas/04_grafo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from aphelion import config
from aphelion.grafo import construccion, entidades as ent, relaciones as rel
from aphelion.indice.chunking import dividir_en_oraciones


def leer_fragmentos(limite: int | None) -> list[dict]:
    """Los fragmentos del corpus, de donde estén.

    `trabajo/fragmentos.jsonl` es la fuente natural, pero no se versiona y en una
    máquina que solo tenga la entrega no existe. La metadata del índice principal
    contiene exactamente los mismos registros —es lo que `04_indexar` escribió a
    partir de ellos—, así que sirve igual y evita reconstruir el corpus entero
    para dibujar un grafo.
    """
    candidatas = (
        config.FRAGMENTOS,
        config.BASE_VECTORIAL / f"encoder_{config.ENCODER_PRINCIPAL}" / "metadata.jsonl",
    )
    for ruta in candidatas:
        if not ruta.exists():
            continue
        registros: list[dict] = []
        with ruta.open(encoding="utf-8") as fh:
            for linea in fh:
                if not linea.strip():
                    continue
                registros.append(json.loads(linea))
                if limite and len(registros) >= limite:
                    break
        print(f"fragmentos: {len(registros):,} de {ruta.name}")
        return registros

    raise FileNotFoundError(
        "no hay fragmentos: ejecuta antes scripts/etapas/03_fragmentar.py"
    )


def reconocer(fragmentos: list[dict], backend_nombre: str, modelo: str | None) -> dict:
    """NER sobre cada fragmento, con aviso de progreso cada minuto de trabajo."""
    backend = ent.cargar_backend(backend_nombre, modelo)
    menciones: dict[str, list[ent.Mencion]] = {}
    t0 = time.time()

    for i, fragmento in enumerate(fragmentos, start=1):
        encontradas = ent.menciones_de_fragmento(fragmento, backend)
        if encontradas:
            menciones[fragmento["chunk_id"]] = encontradas
        if i % 2000 == 0:
            ritmo = i / max(time.time() - t0, 1e-9)
            restante = (len(fragmentos) - i) / max(ritmo, 1e-9) / 60
            print(f"  {i:,}/{len(fragmentos):,}  {ritmo:.0f}/s  faltan ~{restante:.0f} min")

    print(f"NER en {(time.time() - t0) / 60:.1f} min")
    return menciones


def extraer_relaciones(fragmentos: list[dict], menciones: dict, clave_a_id: dict) -> list:
    tripletas: list[rel.Tripleta] = []
    for fragmento in fragmentos:
        del_fragmento = menciones.get(fragmento["chunk_id"])
        if not del_fragmento or len(del_fragmento) < 2:
            continue
        texto = fragmento["texto"]
        oraciones = dividir_en_oraciones(texto, fragmento.get("idioma") or "es")
        tripletas.extend(rel.extraer(texto, del_fragmento, oraciones, clave_a_id))
    return tripletas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ner", default="onnx", choices=("onnx", "falso"))
    ap.add_argument("--modelo", help="repositorio del modelo de NER, si no el de por defecto")
    ap.add_argument("--limite", type=int, help="usa solo los primeros N fragmentos")
    ap.add_argument("--rehacer-ner", action="store_true", help="ignora trabajo/entidades.jsonl")
    ap.add_argument(
        "--agrupar-entidades",
        action="store_true",
        help="fusiona nombres equivalentes entre idiomas con el encoder principal",
    )
    ap.add_argument("--destino", type=Path, default=config.GRAFO)
    args = ap.parse_args()

    fragmentos = leer_fragmentos(args.limite)

    # El backend va en el nombre del archivo: una comprobación con `--ner falso`
    # no puede dejar una caché que la corrida real dé por buena.
    cache = config.ruta_entidades(args.ner)

    if cache.exists() and not args.rehacer_ner:
        menciones = ent.leer_cache(cache)
        print(f"entidades en caché ({cache.name}): {len(menciones):,} fragmentos")
    else:
        try:
            menciones = reconocer(fragmentos, args.ner, args.modelo)
        except ImportError:
            # No se devuelve error a propósito. Esta etapa corre dentro del
            # pipeline y el grafo es el componente **bonus**: tumbar aquí la
            # corrida se llevaría por delante `05_empaquetar` y `06_verificar`,
            # que es la que comprueba lo único eliminatorio del reto. Es el mismo
            # trato que `05_empaquetar` da al PDF del informe: avisar fuerte y
            # seguir.
            print()
            print("  PENDIENTE: falta el backend de NER, no hay grafo (§7, bonus).")
            print("  Instálalo con:  uv sync --extra grafo")
            print("  o córrelo aislado:")
            print("      uv run --isolated --with gliner2-onnx python scripts/etapas/04_grafo.py")
            print("  El resto de la entrega no depende de esto.")
            return 0
        ent.escribir_cache(menciones, cache)
        print(f"-> {cache}")

    todas = [m for lista in menciones.values() for m in lista]
    print(f"menciones: {len(todas):,}")

    equivalencias: dict[str, str] = {}
    if args.agrupar_entidades:
        # Se agrupa con el mismo encoder que construyó el índice: es cross-lingüe
        # por construcción, ya está anclado en la entrega y no añade ninguna
        # licencia nueva que declarar en el informe.
        from aphelion.indice import encoders

        claves = sorted({m.clave for m in todas if m.clave})
        print(f"agrupando {len(claves):,} nombres con {config.ENCODER_PRINCIPAL}…")
        encoder = encoders.cargar(config.ENCODER_PRINCIPAL)
        equivalencias = ent.agrupar_por_embedding(claves, encoder.codificar_consultas)
        print(f"  {len(equivalencias):,} nombres fusionados en otro")

    entidades, clave_a_id = ent.canonicalizar(todas, equivalencias)
    n_documentos = len({f["doc_id"] for f in fragmentos})
    admitidas = construccion.entidades_admitidas(entidades, n_documentos)
    print(f"entidades: {len(entidades):,} -> {len(admitidas):,} tras podar")

    tripletas = extraer_relaciones(fragmentos, menciones, clave_a_id)
    print(f"tripletas: {len(tripletas):,}")

    grafo = construccion.construir(
        fragmentos, menciones, admitidas, clave_a_id, tripletas
    )
    destino = construccion.exportar(grafo, args.destino)

    resumen = construccion.resumen(grafo)
    print("\ngrafo:")
    for clave, valor in resumen.items():
        print(f"  {clave:12} {valor:>9,}")
    print(f"\n-> {destino}  ({destino.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
