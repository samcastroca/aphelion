"""Arma el pool de fragmentos a juzgar y lo convierte en ground truth.

Complementa a `pool_anotacion.py`, que reparte el pool en CSV entre cuatro
personas. Este emite **un solo archivo** con todo el pool, pensado para una
pasada de anotación seguida, y devuelve los juicios al mismo formato de CSV por
anotador que ya consume `pool_anotacion.py --consolidar`. Así los juicios de
esta pasada conviven con los humanos, y el cálculo de kappa entre anotadores
sigue funcionando sin tocarlo.

**El pool sale de la unión del top-N de cada encoder por separado**, no de la
salida fusionada. Anotar solo lo que el sistema devuelve hoy sesga cualquier
comparación posterior: una configuración que sacara a la superficie fragmentos
nuevos los vería como no anotados —relevancia 0— y quedaría penalizada por
mejorar.

**Sobre quién juzga.** La §8.3 prohíbe los modelos generativos en la
recuperación, y esta anotación no está en la recuperación: el ground truth es un
instrumento de medida, nunca entra al índice ni al `resultados.jsonl` que se
entrega. Un modelo puede rellenar la columna. Lo que no se puede saltar es el
control: unos juicios que nadie contrastó miden el criterio de quien anotó, no
la calidad del sistema. Por eso `--muestra-humana` separa un subconjunto para
que lo juzgue una persona en paralelo, y la consolidación reporta el kappa
entre ambos. Con kappa bajo, el ground truth no sirve para elegir configuración.

Uso:
    uv run python scripts/analisis/pool_juicios.py --generar
    uv run python scripts/analisis/pool_juicios.py --lote 1     # imprime un lote
    uv run python scripts/analisis/pool_juicios.py --consolidar # juicios -> CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas
from aphelion.indice import vectores

POOL = config.TRABAJO / "pool_juicios.jsonl"
JUICIOS = config.TRABAJO / "juicios"  # un JSONL por lote
CACHE_RANKINGS = config.TRABAJO / "rankings.jsonl"

# Cuántos candidatos aporta cada encoder al pool. 15 cubre con margen el top-10
# que evalúa NDCG@10 en las tres corridas que se comparan (cada encoder por
# separado y la fusión): un fragmento que ninguno de los dos coloca entre sus
# quince primeros no llega al top-10 de ninguna de ellas.
PROFUNDIDAD_POOL = 15

# Grado a partir del cual un fragmento hace relevante a su documento. Mismo
# valor y mismo criterio que `pool_anotacion.py`: 'parcial' (1) no basta.
RELEVANTE = 2.0

# Consultas por lote de revisión. Con quince candidatos por encoder salen unos
# 20-25 fragmentos únicos por consulta; cinco consultas por lote es lo que se
# puede juzgar de una sentada sin perder el criterio por el camino.
CONSULTAS_POR_LOTE = 5


def cargar_rankings(ruta: Path) -> dict[str, dict[str, list[dict]]]:
    if not ruta.exists():
        raise FileNotFoundError(
            f"no existe {ruta}. Ejecuta antes scripts/analisis/comparar_encoders.py"
        )
    cache: dict[str, dict[str, list[dict]]] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            cache[registro["query_id"]] = registro["rankings"]
    return cache


def indice_metadata(base: Path) -> dict[str, dict]:
    """chunk_id -> metadata, leído de un solo índice.

    Los dos encoders indexan exactamente los mismos fragmentos en el mismo
    orden, así que basta el primero: lo que cambia entre ellos son los vectores,
    no la metadata.
    """
    nombres = vectores.encoders_disponibles(base)
    if not nombres:
        raise FileNotFoundError(f"no hay índices en {base}")

    ruta = vectores.carpeta_encoder(nombres[0], base) / "metadata.jsonl"
    por_chunk: dict[str, dict] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            por_chunk[registro["chunk_id"]] = registro
    return por_chunk


def generar(base: Path, profundidad: int, destino: Path) -> int:
    cache = cargar_rankings(CACHE_RANKINGS)
    preguntas = {p.query_id: p for p in mod_consultas.cargar()}
    print(f"resolviendo metadata de {base} ...", flush=True)
    meta = indice_metadata(base)
    print(f"  {len(meta):,} fragmentos en el índice")

    filas: list[dict] = []
    for query_id in sorted(cache):
        pregunta = preguntas.get(query_id)
        if pregunta is None:
            continue

        # Se conserva la posición que el fragmento ocupó en cada encoder: es lo
        # que después permite explicar por qué entró al pool y comparar de dónde
        # sale la evidencia que cada uno encuentra.
        posiciones: dict[str, dict[str, float]] = {}
        for encoder, ranking in cache[query_id].items():
            for posicion, entrada in enumerate(ranking[:profundidad], start=1):
                registro = posiciones.setdefault(entrada["chunk_id"], {})
                registro[encoder] = posicion
                registro[f"sim_{encoder}"] = entrada["sim"]

        for chunk_id, marcas in posiciones.items():
            fragmento = meta.get(chunk_id)
            if fragmento is None:
                continue
            filas.append(
                {
                    "query_id": query_id,
                    "pregunta": pregunta.texto,
                    "fenomeno_consulta": pregunta.fenomeno,
                    "chunk_id": chunk_id,
                    "doc_id": fragmento["doc_id"],
                    "fuente": fragmento["fuente"],
                    "formato": fragmento.get("formato", ""),
                    "fenomeno_doc": fragmento.get("fenomeno"),
                    "idioma": fragmento.get("idioma", ""),
                    "posiciones": {k: v for k, v in marcas.items() if not k.startswith("sim_")},
                    "similitudes": {k[4:]: v for k, v in marcas.items() if k.startswith("sim_")},
                    "texto": fragmento["texto"],
                }
            )

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for fila in filas:
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")

    por_query: dict[str, int] = {}
    for fila in filas:
        por_query[fila["query_id"]] = por_query.get(fila["query_id"], 0) + 1
    solo_uno = sum(1 for f in filas if len(f["posiciones"]) == 1)

    print(f"\npool: {len(filas):,} fragmentos sobre {len(por_query)} consultas")
    print(f"  por consulta: min {min(por_query.values())}, "
          f"max {max(por_query.values())}, "
          f"media {sum(por_query.values()) / len(por_query):.1f}")
    print(f"  aportados por un solo encoder: {solo_uno:,} ({solo_uno / len(filas):.1%})")
    print(f"  lotes de {CONSULTAS_POR_LOTE} consultas: "
          f"{(len(por_query) + CONSULTAS_POR_LOTE - 1) // CONSULTAS_POR_LOTE}")
    print(f"\n-> {destino}")
    return 0


def lote(destino: Path, numero: int, max_chars: int) -> int:
    """Imprime un lote de consultas del pool, listo para juzgar."""
    with destino.open(encoding="utf-8") as fh:
        filas = [json.loads(l) for l in fh if l.strip()]

    query_ids = sorted({f["query_id"] for f in filas})
    inicio = (numero - 1) * CONSULTAS_POR_LOTE
    del_lote = query_ids[inicio : inicio + CONSULTAS_POR_LOTE]
    if not del_lote:
        print(f"el lote {numero} está fuera de rango ({len(query_ids)} consultas)")
        return 1

    for query_id in del_lote:
        propias = [f for f in filas if f["query_id"] == query_id]
        print(f"\n{'=' * 78}")
        print(f"{query_id}  (fenómeno {propias[0]['fenomeno_consulta']})")
        print(f"{propias[0]['pregunta']}")
        print(f"{'=' * 78}")
        for fila in sorted(
            propias, key=lambda f: min(f["posiciones"].values())
        ):
            marcas = " ".join(f"{k}={v}" for k, v in sorted(fila["posiciones"].items()))
            texto = " ".join(fila["texto"].split())
            if len(texto) > max_chars:
                texto = texto[:max_chars] + " […]"
            print(f"\n--- {fila['chunk_id']}  [{marcas}]  {fila['fuente']} "
                  f"(f{fila['fenomeno_doc']}, {fila['idioma']}, {fila['formato']})")
            print(texto)
    return 0


def _archivos_de_juicios(juicios: Path) -> list[Path]:
    """Acepta un archivo o un directorio con un JSONL por lote.

    Juzgar 1.383 fragmentos de una sentada no es realista, así que los juicios
    se escriben por lotes y se unen al consolidar.
    """
    if juicios.is_dir():
        return sorted(juicios.glob("*.jsonl"))
    return [juicios] if juicios.exists() else []


def consolidar(pool: Path, juicios: Path, destino_dir: Path, nombre: str) -> int:
    """Convierte los juicios en el CSV por anotador que ya lee pool_anotacion.py."""
    archivos = _archivos_de_juicios(juicios)
    if not archivos:
        print(f"no hay juicios en {juicios}")
        return 1

    with pool.open(encoding="utf-8") as fh:
        filas = {(f["query_id"], f["chunk_id"]): f for f in (json.loads(l) for l in fh if l.strip())}

    emitidos: dict[tuple[str, str], float] = {}
    docs_forzados: set[tuple[str, str]] = set()  # (query_id, doc_id)
    for archivo in archivos:
        with archivo.open(encoding="utf-8") as fh:
            for linea in fh:
                if not linea.strip():
                    continue
                juicio = json.loads(linea)
                clave = (juicio["query_id"], juicio["chunk_id"])
                if clave not in filas:
                    print(f"  aviso: {clave} no está en el pool, se ignora")
                    continue
                emitidos[clave] = float(juicio["relevancia"])
                # `doc` marca el documento como relevante aunque su fragmento
                # puntúe bajo. Hace falta porque el reto usa dos claves de
                # emparejamiento distintas (§10.2.1): los fragmentos se juzgan
                # por su texto y los documentos por su fuente. Un fragmento que
                # solo trae el título del artículo no es evidencia —y el jurado
                # no lo puntuaría— pero el documento del que sale sí puede ser
                # relevante. Sin este canal, los 80 artículos del CEEEP que
                # entraron al índice sin su `abstract` quedarían inalcanzables
                # para el F1@3 por un defecto de extracción, no por el ranking.
                if juicio.get("doc"):
                    docs_forzados.add((juicio["query_id"], filas[clave]["doc_id"]))
    print(f"juicios leídos de {len(archivos)} archivo(s)")

    faltan = set(filas) - set(emitidos)
    if faltan:
        sin_juzgar = sorted({q for q, _ in faltan})
        print(f"  {len(faltan)} fragmentos sin juzgar en {len(sin_juzgar)} consultas")
        print(f"  consultas afectadas: {sin_juzgar[:10]}")

    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / f"{nombre}.csv"
    campos = [
        "query_id", "pregunta", "encoder", "rango", "similitud",
        "doc_id", "fuente", "chunk_id", "idioma", "texto", "relevancia", "notas",
    ]
    with destino.open("w", encoding="utf-8-sig", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=campos)
        escritor.writeheader()
        for clave, grado in sorted(emitidos.items()):
            fila = filas[clave]
            escritor.writerow(
                {
                    "query_id": fila["query_id"],
                    "pregunta": fila["pregunta"],
                    "encoder": "+".join(sorted(fila["posiciones"])),
                    "rango": min(fila["posiciones"].values()),
                    "similitud": max(fila["similitudes"].values()),
                    "doc_id": fila["doc_id"],
                    "fuente": fila["fuente"],
                    "chunk_id": fila["chunk_id"],
                    "idioma": fila["idioma"],
                    "texto": fila["texto"],
                    "relevancia": grado,
                    "notas": "",
                }
            )

    reparto: dict[float, int] = {}
    for grado in emitidos.values():
        reparto[grado] = reparto.get(grado, 0) + 1
    print(f"\n{len(emitidos):,} juicios -> {destino}")
    print(f"  reparto: {dict(sorted(reparto.items()))}")

    escribir_ground_truth(filas, emitidos, docs_forzados, config.GROUND_TRUTH)
    return 0


def escribir_ground_truth(
    filas: dict[tuple[str, str], dict],
    emitidos: dict[tuple[str, str], float],
    docs_forzados: set[tuple[str, str]],
    destino: Path,
) -> Path:
    """Escribe `ground_truth.jsonl` directamente, con las dos claves separadas.

    No se pasa por `pool_anotacion.py --consolidar` porque ese camino deriva la
    relevancia del documento del grado del fragmento (>= 2), y aquí hacen falta
    las dos por separado: un fragmento de título puntúa 0 o 1 y su documento
    puede seguir siendo relevante. El CSV se emite igual, para que el kappa
    contra anotadores humanos siga calculándose con la maquinaria que ya existe.
    """
    fragmentos: dict[str, dict[str, float]] = {}
    documentos: dict[str, set[str]] = {}

    for (query_id, chunk_id), grado in emitidos.items():
        fragmentos.setdefault(query_id, {})[chunk_id] = grado
        if grado >= RELEVANTE:
            documentos.setdefault(query_id, set()).add(
                filas[(query_id, chunk_id)]["doc_id"]
            )

    for query_id, doc_id in docs_forzados:
        documentos.setdefault(query_id, set()).add(doc_id)

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for query_id in sorted(fragmentos):
            fh.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "fragmentos": fragmentos[query_id],
                        "documentos": sorted(documentos.get(query_id, ())),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # El texto de cada fragmento juzgado, aparte. Sin esto los juicios solo
    # sirven para la fragmentación que los originó: al cambiar el tamaño de chunk
    # los `chunk_id` dejan de existir y hay que emparejar por texto, que es
    # además como lo hará el jurado (§10.2.1).
    juzgados = {chunk_id for _, chunk_id in emitidos}
    with config.GROUND_TRUTH_TEXTOS.open("w", encoding="utf-8") as fh:
        for clave in sorted(emitidos):
            chunk_id = clave[1]
            if chunk_id not in juzgados:
                continue
            juzgados.discard(chunk_id)  # una sola línea por chunk_id
            fh.write(
                json.dumps(
                    {"chunk_id": chunk_id, "texto": filas[clave]["texto"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    relevantes = sum(len(d) for d in documentos.values())
    con_doc = sum(1 for q in fragmentos if documentos.get(q))
    print(f"textos juzgados -> {config.GROUND_TRUTH_TEXTOS}")
    print(f"\nground truth -> {destino}")
    print(f"  {len(fragmentos)} consultas, {relevantes} documentos relevantes")
    print(f"  {len(docs_forzados)} marcados por título (documento sí, fragmento no)")
    if con_doc < len(fragmentos):
        print(f"  aviso: {len(fragmentos) - con_doc} consultas sin ningún documento "
              f"relevante; su F1@3 será 0 por construcción")
    return destino


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generar", action="store_true", help="arma el pool desde los rankings")
    ap.add_argument("--lote", type=int, help="imprime el lote N para juzgarlo")
    ap.add_argument("--consolidar", action="store_true", help="juicios -> CSV de anotador")
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--profundidad", type=int, default=PROFUNDIDAD_POOL)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--juicios", type=Path, default=JUICIOS)
    ap.add_argument("--nombre", default="anotador_modelo", help="nombre del CSV de salida")
    ap.add_argument("--max-chars", type=int, default=700, help="recorte al imprimir")
    args = ap.parse_args()

    if args.generar:
        return generar(args.base, args.profundidad, args.pool)
    if args.lote:
        return lote(args.pool, args.lote, args.max_chars)
    if args.consolidar:
        return consolidar(args.pool, args.juicios, config.ANOTACION, args.nombre)

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
