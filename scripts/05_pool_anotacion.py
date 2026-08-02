"""Genera el pool de candidatos que el equipo debe anotar.

El ground truth oficial no es público. Para poder medir, se anota a mano sobre las
50 consultas reales.

El pool se construye uniendo el top-N de **cada encoder por separado**, no del
sistema fusionado. Esto mitiga el sesgo de evaluar únicamente lo que la
configuración actual ya encuentra: si solo se anotara la salida final, cualquier
cambio que recuperase documentos nuevos aparecería como ruido no anotado y se
penalizaría injustamente.

Produce un CSV por anotador, repartido de forma que cada consulta tenga un
responsable y una fracción quede duplicada para medir acuerdo entre anotadores.

Los CSV y el ground truth resultante viven en `datos/`, versionados: son juicios
emitidos a mano, el artefacto más caro del proyecto y el único que no se
regenera ejecutando nada.

Sobre quién anota: cualquiera que lea la consulta y el fragmento puede rellenar
la columna, incluido un asistente. Lo que no se puede saltar es el control: el
solape entre anotadores está para medir el acuerdo, y si parte del pool lo juzga
un modelo, ese solape debe incluir a personas. Un ground truth que nadie
contrastó mide el criterio de quien anotó, no la calidad del sistema.

Uso:
    uv run python scripts/05_pool_anotacion.py --anotadores 4 --top 20
    # ... el equipo rellena la columna 'relevancia' en los CSV ...
    uv run python scripts/05_pool_anotacion.py --consolidar
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from aphelion import config, consultas as mod_consultas, encoders, recuperacion, vectores


RELEVANTE = 2.0  # 'parcial' no cuenta como documento relevante


def kappa_cohen(pares: list[tuple[float, float]]) -> float | None:
    """Acuerdo entre dos anotadores sobre los mismos fragmentos, corregido por azar.

    El solape entre anotadores existe justamente para poder calcular esto. Si el
    acuerdo es bajo, el ground truth no sirve para comparar configuraciones:
    estaríamos midiendo el criterio de quien anotó, no la calidad del sistema.
    Por debajo de 0,61 el acuerdo deja de considerarse sustancial.
    """
    if not pares:
        return None

    n = len(pares)
    categorias = sorted({v for par in pares for v in par})
    observado = sum(1 for a, b in pares if a == b) / n

    esperado = 0.0
    for c in categorias:
        p_a = sum(1 for a, _ in pares if a == c) / n
        p_b = sum(1 for _, b in pares if b == c) / n
        esperado += p_a * p_b

    if esperado >= 1.0:
        return 1.0  # todos coincidieron en una sola categoría
    return (observado - esperado) / (1 - esperado)


def medir_acuerdo(por_anotador: dict[str, dict[str, float]]) -> None:
    """Reporta el acuerdo de cada par de anotadores con fragmentos en común."""
    nombres = sorted(por_anotador)
    reportado = False

    for i, a in enumerate(nombres):
        for b in nombres[i + 1 :]:
            comunes = set(por_anotador[a]) & set(por_anotador[b])
            if len(comunes) < 20:
                continue
            pares = [(por_anotador[a][c], por_anotador[b][c]) for c in sorted(comunes)]
            k = kappa_cohen(pares)
            if k is None:
                continue
            if not reportado:
                print("\nacuerdo entre anotadores:")
                reportado = True
            aviso = "" if k >= 0.61 else "   <- por debajo de lo aceptable"
            print(f"  {a} vs {b}: kappa {k:.2f} sobre {len(comunes)} fragmentos{aviso}")

    if not reportado:
        print("\nsin solape suficiente para medir acuerdo entre anotadores")


def consolidar(origen: Path, destino: Path) -> int:
    """Convierte los CSV anotados en el ground_truth.jsonl que lee 04_evaluar.

    Cuando dos anotadores juzgan la misma consulta se toma el máximo, no el
    promedio: si alguien reconoció un fragmento como relevante, lo es. Promediar
    castigaría la evidencia encontrada por una sola persona.
    """
    if not origen.exists():
        print(f"no existe {origen}. Genera el pool primero.")
        return 1

    csvs = sorted(origen.glob("*.csv"))
    if not csvs:
        print(f"no hay CSV en {origen}")
        return 1

    fragmentos: dict[str, dict[str, float]] = {}
    documentos: dict[str, set[str]] = {}
    por_anotador: dict[str, dict[str, float]] = {}
    juicios, sin_anotar = 0, 0

    for ruta in csvs:
        propios: dict[str, float] = {}
        with ruta.open(encoding="utf-8-sig", newline="") as fh:
            for fila in csv.DictReader(fh):
                bruto = (fila.get("relevancia") or "").strip()
                if not bruto:
                    sin_anotar += 1
                    continue
                try:
                    grado = float(bruto)
                except ValueError:
                    print(f"  {ruta.name}: relevancia ilegible {bruto!r}, se ignora")
                    continue

                juicios += 1
                q = fila["query_id"]
                previo = fragmentos.setdefault(q, {}).get(fila["chunk_id"], 0.0)
                fragmentos[q][fila["chunk_id"]] = max(previo, grado)
                propios[f"{q}|{fila['chunk_id']}"] = grado
                if grado >= RELEVANTE:
                    documentos.setdefault(q, set()).add(fila["doc_id"])
        por_anotador[ruta.stem] = propios

    if not juicios:
        print("ningún CSV tiene la columna 'relevancia' rellenada")
        return 1

    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for q in sorted(fragmentos):
            fh.write(
                json.dumps(
                    {
                        "query_id": q,
                        "fragmentos": fragmentos[q],
                        "documentos": sorted(documentos.get(q, ())),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"{juicios:,} juicios sobre {len(fragmentos)} consultas -> {destino}")
    if juicios < 500:
        print(f"  {juicios} juicios es poco: por debajo de ~500 las comparaciones")
        print("  entre configuraciones dejan de ser fiables")
    medir_acuerdo(por_anotador)
    if sin_anotar:
        print(f"  {sin_anotar:,} filas siguen sin anotar")
    faltan = 50 - len(fragmentos)
    if faltan > 0:
        print(f"  {faltan} consultas sin ningún juicio: quedarán fuera del promedio")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="candidatos por encoder")
    ap.add_argument("--anotadores", type=int, default=4)
    ap.add_argument("--solape", type=float, default=0.2, help="fracción duplicada")
    ap.add_argument("--salida", type=Path, default=config.ANOTACION)
    ap.add_argument(
        "--consolidar",
        action="store_true",
        help="convierte los CSV ya anotados en datos/ground_truth.jsonl",
    )
    args = ap.parse_args()

    if args.consolidar:
        return consolidar(args.salida, config.GROUND_TRUTH)

    disponibles = vectores.encoders_disponibles()
    if not disponibles:
        print("no hay índices construidos; ejecuta antes scripts/03_indexar.py")
        return 1

    indices = recuperacion.cargar_indices(disponibles)
    preguntas = mod_consultas.cargar()
    args.salida.mkdir(parents=True, exist_ok=True)

    filas: list[dict] = []
    for pregunta in preguntas:
        vistos: set[str] = set()
        for nombre, indice in indices.items():
            vector = encoders.cargar(nombre).codificar_consultas([pregunta.texto])[0]
            for posicion, (idx, similitud) in enumerate(
                indice.buscar(vector, args.top), start=1
            ):
                meta = indice.meta(idx)
                if meta["chunk_id"] in vistos:
                    continue
                vistos.add(meta["chunk_id"])
                filas.append(
                    {
                        "query_id": pregunta.query_id,
                        "pregunta": pregunta.texto,
                        "encoder": nombre,
                        "rango": posicion,
                        "similitud": round(similitud, 4),
                        "doc_id": meta["doc_id"],
                        "fuente": meta["fuente"],
                        "chunk_id": meta["chunk_id"],
                        "idioma": meta.get("idioma", ""),
                        # Sin truncar: un fragmento de 512 tokens pasa de 2.000
                        # caracteres, y juzgar relevancia sobre un texto cortado
                        # produce juicios sobre algo que no es lo que se indexó.
                        "texto": meta["texto"],
                        # columnas a rellenar por el anotador
                        "relevancia": "",  # 0 = no, 1 = parcial, 2 = relevante
                        "notas": "",
                    }
                )

    if not filas:
        print("el pool quedó vacío")
        return 1

    # Reparto por consulta, con una fracción duplicada para medir acuerdo.
    orden = [p.query_id for p in preguntas]
    asignacion = {query_id: [i % args.anotadores] for i, query_id in enumerate(orden)}

    for query_id in orden[: int(len(orden) * args.solape)]:
        segundo = (asignacion[query_id][0] + 1) % args.anotadores
        if segundo not in asignacion[query_id]:
            asignacion[query_id].append(segundo)

    campos = list(filas[0].keys())
    for anotador in range(args.anotadores):
        suyas = {q for q, duenos in asignacion.items() if anotador in duenos}
        propias = [f for f in filas if f["query_id"] in suyas]
        destino = args.salida / f"anotador_{anotador + 1}.csv"
        with destino.open("w", encoding="utf-8-sig", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(propias)
        print(
            f"anotador {anotador + 1}: {len(suyas):>2} consultas, "
            f"{len(propias):>4} filas -> {destino}"
        )

    print(f"\ntotal de juicios a emitir: {len(filas):,}")
    print("Rellena la columna 'relevancia': 0 = no relevante, 1 = parcial, 2 = relevante.")
    print("Al terminar: uv run python scripts/05_pool_anotacion.py --consolidar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
