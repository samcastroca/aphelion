"""Evalúa uno o varios `resultados.jsonl` contra el ground truth y los compara.

Calcula las dos métricas del reto —NDCG@10 sobre fragmentos y F1@3 sobre
documentos— con las fórmulas de la §10.2, y ordena las corridas por Conteo de
Borda, que es como el reto ordena a los equipos.

**Los documentos se emparejan por `fuente`, no por `doc_id`.** La §10.2.1 lo dice
explícitamente: el `doc_id` es un identificador interno de cada equipo y el
emparejamiento con el ground truth se hace por el nombre del archivo original.
El corpus tiene 59 nombres estandarizados repetidos en 186 archivos, así que la
diferencia no es teórica: dos `doc_id` distintos con la misma `fuente` son, para
el jurado, el mismo documento. Con `--por-doc-id` se mide a la antigua, que
sirve para ver cuánto cambia.

Los fragmentos sí se emparejan por `chunk_id`, y eso es correcto **aquí**: el
ground truth interno se anotó sobre estos mismos fragmentos. El jurado juzgará
por el texto (§10.2.1), porque sus anotaciones no conocen nuestro chunking.

Uso:
    uv run python scripts/analisis/comparar.py entrega/resultados.jsonl
    uv run python scripts/analisis/comparar.py trabajo/resultados_*.jsonl --detalle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aphelion import config
from aphelion.evaluacion import metricas
from aphelion.indice import vectores

MAPA_FUENTES = config.TRABAJO / "doc_id_a_fuente.json"


def mapa_doc_a_fuente(base: Path, cache: Path = MAPA_FUENTES) -> dict[str, str]:
    """doc_id -> fuente, leído del índice y cacheado.

    Recorrer los 141 MB de `metadata.jsonl` cuesta unos segundos y el resultado
    son unos pocos miles de pares, así que se guarda: comparar cinco corridas no
    debería releer el índice cinco veces.
    """
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    nombres = vectores.encoders_disponibles(base)
    if not nombres:
        raise FileNotFoundError(f"no hay índices en {base}")

    ruta = vectores.carpeta_encoder(nombres[0], base) / "metadata.jsonl"
    mapa: dict[str, str] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            mapa.setdefault(registro["doc_id"], registro["fuente"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapa, ensure_ascii=False), encoding="utf-8")
    return mapa


def evaluar_corrida(
    resultados: list[dict],
    juicios: dict[str, metricas.Juicio],
    mapa: dict[str, str] | None,
) -> dict:
    """Como `metricas.evaluar`, pero resolviendo documentos a `fuente`.

    Se apoya en las mismas primitivas —`ndcg_at_k`, `f1_at_k`— para no tener dos
    implementaciones de las fórmulas que deciden cada ajuste.
    """
    ndcgs: list[float] = []
    f1s: list[float] = []
    por_consulta: dict[str, dict] = {}

    def resolver(doc_id: str) -> str:
        return mapa.get(doc_id, doc_id) if mapa else doc_id

    for resultado in resultados:
        query_id = resultado["query_id"]
        juicio = juicios.get(query_id)
        if juicio is None:
            continue

        obtenidas = [
            juicio.fragmentos.get(f["chunk_id"], 0.0)
            for f in resultado.get("fragments", [])
        ]
        ndcg = metricas.ndcg_at_k(obtenidas, list(juicio.fragmentos.values()), 10)

        devueltos = [resolver(d["doc_id"]) for d in resultado.get("documents", [])]
        relevantes = {resolver(d) for d in juicio.documentos}
        f1 = metricas.f1_at_k(devueltos, relevantes, 3)

        ndcgs.append(ndcg)
        f1s.append(f1)
        por_consulta[query_id] = {
            "ndcg@10": round(ndcg, 4),
            "f1@3": round(f1, 4),
            "fenomeno": config.fenomeno_de_consulta(query_id),
        }

    return {
        "consultas_evaluadas": len(ndcgs),
        "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
        "f1@3": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "ic_ndcg@10": tuple(round(v, 4) for v in metricas.intervalo_bootstrap(ndcgs)),
        "ic_f1@3": tuple(round(v, 4) for v in metricas.intervalo_bootstrap(f1s)),
        "por_consulta": por_consulta,
    }


def puntos_borda(resumenes: list[dict]) -> list[int]:
    """Conteo de Borda sobre las dos métricas, como en la §11.2 del reto."""
    n = len(resumenes)
    puntos = [0] * n
    for clave in ("ndcg@10", "f1@3"):
        orden = sorted(range(n), key=lambda i: resumenes[i][clave], reverse=True)
        for posicion, i in enumerate(orden, start=1):
            puntos[i] += n - posicion
    return puntos


def por_fenomeno(resumen: dict) -> dict[int, tuple[float, float, int]]:
    agrupado: dict[int, list[tuple[float, float]]] = {}
    for datos in resumen["por_consulta"].values():
        agrupado.setdefault(datos["fenomeno"], []).append(
            (datos["ndcg@10"], datos["f1@3"])
        )
    return {
        fenomeno: (
            sum(n for n, _ in pares) / len(pares),
            sum(f for _, f in pares) / len(pares),
            len(pares),
        )
        for fenomeno, pares in sorted(agrupado.items())
        if fenomeno is not None
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("resultados", nargs="+", type=Path, help="uno o más .jsonl")
    ap.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH)
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument(
        "--por-doc-id",
        action="store_true",
        help="empareja documentos por doc_id en vez de por fuente (§10.2.1)",
    )
    ap.add_argument("--detalle", action="store_true", help="desglose por consulta")
    args = ap.parse_args()

    if not args.ground_truth.exists():
        print(f"no existe {args.ground_truth}.")
        print("\nEl ground truth se construye así:")
        print("  1. uv run python scripts/analisis/comparar_encoders.py")
        print("  2. uv run python scripts/analisis/pool_juicios.py --generar")
        print("  3. juzgar los lotes y escribir trabajo/juicios_modelo.jsonl")
        print("  4. uv run python scripts/analisis/pool_juicios.py --consolidar")
        print("  5. uv run python scripts/analisis/pool_anotacion.py --consolidar")
        return 1

    juicios = metricas.cargar_juicios(args.ground_truth)
    anotados = sum(len(j.fragmentos) for j in juicios.values())
    print(f"ground truth: {len(juicios)} consultas, {anotados:,} fragmentos juzgados")

    mapa = None if args.por_doc_id else mapa_doc_a_fuente(args.base)
    clave = "doc_id" if args.por_doc_id else "fuente"
    print(f"emparejamiento de documentos: por {clave}")

    resumenes: list[dict] = []
    etiquetas: list[str] = []
    for ruta in args.resultados:
        if not ruta.exists():
            print(f"  aviso: no existe {ruta}, se omite")
            continue
        resultados = metricas.cargar_resultados(ruta)
        resumenes.append(evaluar_corrida(resultados, juicios, mapa))
        etiquetas.append(ruta.stem)

    if not resumenes:
        print("ninguna corrida evaluable")
        return 1

    puntos = puntos_borda(resumenes)
    orden = sorted(range(len(resumenes)), key=lambda i: puntos[i], reverse=True)

    ancho = max(len(e) for e in etiquetas) + 2
    print(f"\n{'corrida':<{ancho}} {'NDCG@10':>9} {'IC 95%':>18} "
          f"{'F1@3':>8} {'IC 95%':>18} {'Borda':>6} {'q':>4}")
    for i in orden:
        r = resumenes[i]
        ic_n = f"[{r['ic_ndcg@10'][0]:.3f}, {r['ic_ndcg@10'][1]:.3f}]"
        ic_f = f"[{r['ic_f1@3'][0]:.3f}, {r['ic_f1@3'][1]:.3f}]"
        print(f"{etiquetas[i]:<{ancho}} {r['ndcg@10']:>9.4f} {ic_n:>18} "
              f"{r['f1@3']:>8.4f} {ic_f:>18} {puntos[i]:>6} "
              f"{r['consultas_evaluadas']:>4}")

    print(f"\n{'corrida':<{ancho}} {'fenómeno':>10} {'NDCG@10':>9} {'F1@3':>8} {'q':>4}")
    for i in orden:
        for fenomeno, (ndcg, f1, n) in por_fenomeno(resumenes[i]).items():
            print(f"{etiquetas[i]:<{ancho}} {fenomeno:>10} {ndcg:>9.4f} {f1:>8.4f} {n:>4}")

    if len(resumenes) > 1:
        mejor = orden[0]
        print(f"\nmejor por Borda: {etiquetas[mejor]}")
        empatadas = [
            etiquetas[i]
            for i in orden[1:]
            if resumenes[i]["ic_ndcg@10"][1] >= resumenes[mejor]["ic_ndcg@10"][0]
            and resumenes[mejor]["ic_ndcg@10"][1] >= resumenes[i]["ic_ndcg@10"][0]
            and resumenes[i]["ic_f1@3"][1] >= resumenes[mejor]["ic_f1@3"][0]
            and resumenes[mejor]["ic_f1@3"][1] >= resumenes[i]["ic_f1@3"][0]
        ]
        if empatadas:
            print(f"  indistinguibles de ella: {', '.join(empatadas)}")
            print("  sus intervalos se solapan en ambas métricas; con este número")
            print("  de consultas no hay evidencia para preferir una sobre otra.")

    if args.detalle:
        print(f"\n{'consulta':<10} " + " ".join(f"{e:>22}" for e in etiquetas))
        for query_id in sorted(juicios):
            fila = f"{query_id:<10} "
            for r in resumenes:
                datos = r["por_consulta"].get(query_id)
                celda = (
                    f"ndcg {datos['ndcg@10']:.2f} f1 {datos['f1@3']:.2f}"
                    if datos
                    else "sin evaluar"
                )
                fila += f"{celda:>22} "
            print(fila)

    return 0


if __name__ == "__main__":
    sys.exit(main())
