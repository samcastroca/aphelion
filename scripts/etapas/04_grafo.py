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

**Backends de NER.** Los dos reales sirven `fastino/gliner2-multi-v1`
(Apache-2.0, 205M), mismos pesos; lo que cambia es el motor:

- `gliner2` (por defecto) es el paquete oficial sobre PyTorch. Coloca el modelo
  en la GPU e infiere por lotes, que es lo que hace que esta etapa dure minutos.
- `onnx` es el export vía `gliner2-onnx`, sin inferencia por lotes y con la
  sesión en CPU salvo que se le pase `--proveedor`. Se conserva porque es el
  único camino a DirectML, la GPU de la máquina de desarrollo.
- `falso` no descarga nada y sirve para comprobar el esquema y la exportación de
  punta a punta antes de gastar GPU.

Es una dependencia opcional: si no está instalada, esta etapa lo dice y no rompe
el resto del pipeline.

Uso:
    uv run python scripts/etapas/04_grafo.py                    # NER real, corpus entero
    uv run python scripts/etapas/04_grafo.py --ner falso        # sin modelo, para probar
    uv run python scripts/etapas/04_grafo.py --limite 2000      # una muestra
    uv run python scripts/etapas/04_grafo.py --lote 64          # más ventanas por lote
    uv run python scripts/etapas/04_grafo.py --agrupar-entidades  # fusión cross-lingüe

    # En la Radeon, por DirectML:
    uv run python scripts/etapas/04_grafo.py --ner onnx --proveedor DmlExecutionProvider
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aphelion import config
from aphelion.grafo import construccion, entidades as ent, relaciones as rel
from aphelion.indice.chunking import dividir_en_oraciones


def ruta_fragmentos() -> Path:
    """De dónde salen los fragmentos.

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
        if ruta.exists():
            return ruta

    raise FileNotFoundError(
        "no hay fragmentos: ejecuta antes scripts/etapas/03_fragmentar.py"
    )


def contar_fragmentos() -> int:
    """Cuántos fragmentos hay, sin parsear ninguno.

    Existe para que se pueda extrapolar el coste de una corrida con `--limite` al
    corpus entero sin cargarlo: `leer_fragmentos` construye un dict por línea, y
    sobre los 147 MB de la metadata eso son varios GB de memoria para averiguar un
    número. Mismo papel que `04_indexar.py --huella`: se pregunta y se sale.
    """
    with ruta_fragmentos().open(encoding="utf-8") as fh:
        return sum(1 for linea in fh if linea.strip())


def leer_fragmentos(limite: int | None) -> list[dict]:
    """Los fragmentos del corpus, hasta `limite` si se pide."""
    ruta = ruta_fragmentos()
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


def reconocer(
    fragmentos: list[dict],
    backend_nombre: str,
    modelo: str | None,
    dispositivo: str,
    lote: int,
    proveedores: list[str] | None,
) -> dict:
    """NER sobre el corpus, por lotes, con aviso de progreso y ritmo."""
    backend = ent.cargar_backend(backend_nombre, modelo, dispositivo, lote, proveedores)
    # La atestación de dispositivo: qué motor quedó dónde, dicho por el backend y
    # no por la petición. Si esta línea dice `cpu` o `CPUExecutionProvider`, la
    # corrida va a tardar horas y es aquí donde hay que parar a mirar.
    donde = getattr(backend, "donde", None)
    if donde:
        print(f"  NER en {donde}", flush=True)
    t0 = time.time()

    def progreso(hechas: int, total: int) -> None:
        transcurrido = max(time.time() - t0, 1e-9)
        ritmo = hechas / transcurrido
        restante = (total - hechas) / max(ritmo, 1e-9) / 60
        print(
            f"  ventanas {hechas:,}/{total:,}  {ritmo:.0f}/s  faltan ~{restante:.1f} min",
            flush=True,
        )

    menciones = ent.reconocer_corpus(fragmentos, backend, lote, progreso)

    desalineadas = getattr(backend, "desalineadas", 0)
    if desalineadas:
        # No es fatal —esas menciones se descartan— pero si el número es grande, el
        # modelo está devolviendo offsets en otra unidad y conviene mirarlo antes
        # de fiarse de la evidencia de las tripletas.
        print(f"  aviso: {desalineadas:,} menciones descartadas por offsets que no cuadran")

    print(f"NER en {(time.time() - t0) / 60:.1f} min")
    return menciones


def _oraciones_de(tarea: tuple[str, str]) -> list[str]:
    """Segmenta un fragmento. A nivel de módulo porque tiene que ser picklable."""
    texto, idioma = tarea
    return dividir_en_oraciones(texto, idioma)


def extraer_relaciones(
    fragmentos: list[dict],
    menciones: dict,
    clave_a_id: dict,
    procesos: int = 1,
) -> list:
    """Las tripletas del corpus.

    **La segmentación va en paralelo y la extracción no.** Emparejar menciones
    contra los patrones es un puñado de regex ya compiladas sobre huecos de menos
    de 60 caracteres: no se nota. Lo que cuesta es `dividir_en_oraciones`, que es
    pysbd y que sobre 64.484 fragmentos es una segunda pasada de decenas de
    minutos —`chunking.py` documenta 321 s para un solo CSV—. Quedaba escondida
    detrás de las 15 horas del NER; con el NER en minutos, era lo siguiente.
    """
    candidatos = [
        f for f in fragmentos
        if len(menciones.get(f["chunk_id"]) or ()) >= 2
    ]
    if not candidatos:
        return []

    tareas = [(f["texto"], f.get("idioma") or "es") for f in candidatos]
    if procesos > 1 and len(candidatos) > procesos:
        with ProcessPoolExecutor(max_workers=procesos) as pool:
            segmentados = list(pool.map(_oraciones_de, tareas, chunksize=64))
    else:
        segmentados = [_oraciones_de(t) for t in tareas]

    tripletas: list[rel.Tripleta] = []
    for fragmento, oraciones in zip(candidatos, segmentados):
        tripletas.extend(
            rel.extraer(
                fragmento["texto"],
                menciones[fragmento["chunk_id"]],
                oraciones,
                clave_a_id,
            )
        )
    return tripletas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ner", default="gliner2", choices=("gliner2", "onnx", "falso"))
    ap.add_argument("--modelo", help="repositorio del modelo de NER, si no el de por defecto")
    ap.add_argument(
        "--dispositivo",
        default="cuda",
        help="dónde corre el NER del backend gliner2: cuda, cpu, cuda:1…",
    )
    ap.add_argument(
        "--lote",
        type=int,
        default=32,
        help="ventanas por lote de inferencia. Subirlo satura mejor la GPU; "
        "bajarlo es lo primero que hay que probar ante un out of memory",
    )
    ap.add_argument(
        "--proveedor",
        action="append",
        help="ejecutor de ONNX Runtime para --ner onnx (repetible). Sin esto la "
        "sesión abre en CPU y la GPU no se toca: DmlExecutionProvider en la "
        "Radeon, CUDAExecutionProvider en NVIDIA con onnxruntime-gpu",
    )
    ap.add_argument(
        "--procesos",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="procesos para segmentar en oraciones al extraer relaciones",
    )
    ap.add_argument("--limite", type=int, help="usa solo los primeros N fragmentos")
    ap.add_argument(
        "--contar",
        action="store_true",
        help="imprime cuántos fragmentos hay y sale, sin construir nada",
    )
    ap.add_argument("--rehacer-ner", action="store_true", help="ignora trabajo/entidades.jsonl")
    ap.add_argument(
        "--agrupar-entidades",
        action="store_true",
        help="fusiona nombres equivalentes entre idiomas con el encoder principal",
    )
    ap.add_argument("--destino", type=Path, default=config.GRAFO)
    args = ap.parse_args()

    if args.contar:
        print(contar_fragmentos())
        return 0

    fragmentos = leer_fragmentos(args.limite)

    # El backend va en el nombre del archivo: una comprobación con `--ner falso`
    # no puede dejar una caché que la corrida real dé por buena.
    cache = config.ruta_entidades(args.ner)

    if cache.exists() and not args.rehacer_ner:
        menciones = ent.leer_cache(cache)
        print(f"entidades en caché ({cache.name}): {len(menciones):,} fragmentos")

        # Una caché dejada por una corrida con `--limite` cubre una parte del
        # corpus, y sin este aviso el grafo saldría construido sobre esa parte sin
        # que nada fallara: el archivo se escribe, la etapa devuelve 0 y solo los
        # conteos del resumen —que nadie mira contra el corpus— lo delatarían. Es
        # el mismo silencio que la ventana de poda vacía, y merece el mismo trato.
        cubiertos = sum(1 for f in fragmentos if f["chunk_id"] in menciones)
        if cubiertos < len(fragmentos) // 2:
            print(
                f"  AVISO: la caché solo cubre {cubiertos:,} de los {len(fragmentos):,} "
                f"fragmentos leídos. Parece de una corrida con --limite.\n"
                f"  Rehazla con --rehacer-ner o el grafo saldrá construido sobre "
                f"esa fracción."
            )
    else:
        try:
            menciones = reconocer(
                fragmentos,
                args.ner,
                args.modelo,
                args.dispositivo,
                args.lote,
                args.proveedor,
            )
        except (ImportError, OSError) as error:
            # No se devuelve error a propósito. Esta etapa corre dentro del
            # pipeline y el grafo es el componente **bonus**: tumbar aquí la
            # corrida se llevaría por delante `05_empaquetar` y `06_verificar`,
            # que es la que comprueba lo único eliminatorio del reto. Es el mismo
            # trato que `05_empaquetar` da al PDF del informe: avisar fuerte y
            # seguir.
            #
            # `OSError` además del `ImportError`: un modelo que no se puede
            # descargar —repositorio inexistente, Hub caído, sin red— llega por
            # ahí, y era el hueco por el que esta etapa sí tumbaba la corrida
            # justo en el caso que este bloque decía cubrir.
            print()
            print(f"  PENDIENTE: no hay NER disponible, no hay grafo (§7, bonus).")
            print(f"  Causa: {type(error).__name__}: {error}")
            print("  Instálalo con:  uv sync --extra grafo")
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

    t0 = time.time()
    tripletas = extraer_relaciones(fragmentos, menciones, clave_a_id, args.procesos)
    print(f"tripletas: {len(tripletas):,}  ({(time.time() - t0) / 60:.1f} min)")

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
