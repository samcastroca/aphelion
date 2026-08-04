"""¿La métrica baja porque no encontramos, o porque no ordenamos?

Son dos causas distintas y se arreglan con cosas distintas. Si el documento
relevante no entra en el pool de candidatos, ninguna fusión, realce ni
agregación posterior lo puede recuperar: hay que tocar el encoder, el chunking o
la profundidad, que es lo caro. Si entra y queda mal colocado, el encoder está
bien y lo que hay que mover es la política de ordenamiento, que es gratis.

Confundir las dos cuesta días de GPU mejorando lo que no falla, así que esto se
mide antes de decidir en qué trabajar.

**Qué imprime.**

- El *recall del pool* a varias profundidades: qué fracción de los documentos
  relevantes llega a estar entre los candidatos. Es el techo de la recuperación.
- La *cobertura de fragmentos*: cuántas consultas tienen al menos un fragmento
  relevante en el pool. Es el techo del NDCG@10.
- El *oráculo*: qué darían las métricas si el pool se reordenara perfectamente,
  usando el ground truth. No es alcanzable —usa las respuestas— pero acota lo
  que puede aportar cualquier mejora del ordenamiento, y la distancia entre lo
  que hay y eso es el margen real.

Uso:
    uv run python scripts/analisis/techo.py
    uv run python scripts/analisis/techo.py --encoders bge-m3 --profundidades 10,50,200
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas, recuperacion, salida as mod_salida
from aphelion.evaluacion import emparejamiento, metricas

RAIZ = Path(__file__).resolve().parents[2]

# Umbral por encima del cual se considera que el pool ya trae lo que hace falta
# y el problema es de ordenamiento. No es redondo por gusto: por debajo de esto
# quedan consultas cuyo documento no está, y esas no las arregla reordenar.
RECALL_SUFICIENTE = 0.90

# Por debajo de esta distancia entre la métrica y su techo no hay nada que ganar
# reordenando, aunque el recall sea alto.
MARGEN_MINIMO = 0.05


def recall(vistos: list[str], relevantes: set[str]) -> float | None:
    """Fracción de `relevantes` que aparece en `vistos`. None si no hay ninguno.

    `vistos` llega con repeticiones —el pool trae fragmentos y varios pueden ser
    del mismo documento— y se deduplica antes de contar, o un documento con
    muchos fragmentos inflaría el recall.
    """
    if not relevantes:
        return None
    return len(set(vistos) & relevantes) / len(relevantes)


def veredicto(recall: float, metrica: float) -> str:
    """Dónde está la culpa: en lo que no se encontró o en cómo se ordenó."""
    if recall - metrica < MARGEN_MINIMO:
        return "sin margen"
    if recall >= RECALL_SUFICIENTE:
        return "ordenamiento"
    return "recuperación"


def cargar_barrido():
    ruta = RAIZ / "scripts" / "analisis" / "barrido_completo.py"
    spec = importlib.util.spec_from_file_location("barrido_completo", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["barrido_completo"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", default=",".join(config.ENCODERS_ENTREGA))
    ap.add_argument("--profundidades", default="10,25,50,100,200")
    ap.add_argument("--chunk", type=int, default=config.CHUNK_PRESUPUESTO)
    ap.add_argument("--solape", type=float, default=config.CHUNK_SOLAPE)
    ap.add_argument("--agregacion", default="top2")
    args = ap.parse_args()

    bc = cargar_barrido()
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    profundidades = [int(p) for p in args.profundidades.split(",")]
    maxima = max(profundidades)

    base = bc.ruta_indices(args.chunk, args.solape)
    faltan = [e for e in encoders if not (base / f"encoder_{e}" / "index.faiss").exists()]
    if faltan:
        print(f"sin índice de {faltan} en {base}")
        print("indexa primero con scripts/analisis/barrido_completo.py")
        return 1

    preguntas = mod_consultas.cargar()
    juicios = metricas.cargar_juicios(config.GROUND_TRUTH)
    empar = emparejamiento.emparejadores(
        juicios, emparejamiento.cargar_textos(config.GROUND_TRUTH_TEXTOS))
    mapa = bc.mapa_doc_a_fuente(base)
    indices = recuperacion.cargar_indices(encoders, base)
    resolver = lambda d: mapa.get(d, d)  # noqa: E731

    # Un encoder a la vez y se suelta: dos modelos large residentes no caben.
    pool: dict[str, dict[str, list]] = defaultdict(dict)
    for nombre in encoders:
        print(f"buscando con {nombre} ...", flush=True)
        buscador = recuperacion.Recuperador({nombre: indices[nombre]})
        for p in preguntas:
            pool[p.query_id].update(buscador.buscar(p, candidatos_por_indice=maxima))
        del buscador
        bc.liberar_encoders()

    print()
    print("=== techo de recuperación: ¿está lo relevante en el pool? ===")
    cabecera = f"{'prof':>6} {'recall doc (unión)':>20}"
    for e in encoders:
        cabecera += f" {e.replace('multilingual-', ''):>16}"
    print(cabecera)
    recall_union: dict[int, float] = {}
    for prof in profundidades:
        union, por_encoder = [], {e: [] for e in encoders}
        for p in preguntas:
            relevantes = {resolver(d) for d in juicios[p.query_id].documentos}
            vistos_union: list[str] = []
            for e in encoders:
                vistos = [resolver(m["doc_id"]) for m, _ in pool[p.query_id][e][:prof]]
                vistos_union += vistos
                r = recall(vistos, relevantes)
                if r is not None:
                    por_encoder[e].append(r)
            r = recall(vistos_union, relevantes)
            if r is not None:
                union.append(r)
        recall_union[prof] = sum(union) / len(union)
        fila = f"{prof:>6} {recall_union[prof]:>20.4f}"
        for e in encoders:
            fila += f" {sum(por_encoder[e]) / len(por_encoder[e]):>16.4f}"
        print(fila)

    print()
    print("=== techo de fragmentos: consultas con algún relevante en el pool ===")
    for prof in profundidades:
        cubiertas = []
        for p in preguntas:
            em = empar[p.query_id]
            if not em.ideal or max(em.ideal) == 0:
                continue
            textos = [m["texto"] for e in encoders for m, _ in pool[p.query_id][e][:prof]]
            cubiertas.append(1.0 if max(em.relevancias(textos), default=0) > 0 else 0.0)
        print(f"  prof {prof:>4}: {sum(cubiertas) / len(cubiertas):.4f}")

    print()
    print("=== oráculo: qué daría reordenar el pool perfectamente ===")
    print("(usa el ground truth, así que no es alcanzable; acota el margen)")
    rec = recuperacion.Recuperador(indices)
    actual = _evaluar_actual(rec, pool, preguntas, encoders, maxima,
                             args.agregacion, juicios, empar, mapa)
    print(f"{'':<28} {'NDCG@10':>9} {'F1@3':>8}")
    print(f"{'lo que hay ahora':<28} {actual['ndcg@10']:>9.4f} {actual['f1@3']:>8.4f}")
    for prof in profundidades:
        o = _evaluar_oraculo(pool, preguntas, encoders, prof, juicios, empar,
                             mapa, resolver)
        print(f"{'oráculo sobre prof ' + str(prof):<28} {o['ndcg@10']:>9.4f} {o['f1@3']:>8.4f}")

    print()
    techo = recall_union[maxima]
    print(f"recall del pool a profundidad {maxima}: {techo:.4f}")
    print(f"F1@3 actual: {actual['f1@3']:.4f}  ->  la culpa es del "
          f"{veredicto(techo, actual['f1@3']).upper()}")
    return 0


def _resultados(preguntas, docs_de, frags_de):
    return [
        {"query_id": p.query_id,
         "documents": [{"rank": i, "doc_id": d}
                       for i, d in enumerate(docs_de(p), start=1)],
         "fragments": frags_de(p)}
        for p in preguntas
    ]


def _evaluar_actual(rec, pool, preguntas, encoders, prof, agregacion,
                    juicios, empar, mapa):
    ordenados = {}
    for p in preguntas:
        rankings = {e: pool[p.query_id][e][:prof] for e in encoders}
        ordenados[p.query_id] = rec.ordenar(rankings, p, agregacion=agregacion)
    return metricas.evaluar_textual(
        _resultados(preguntas,
                    lambda p: ordenados[p.query_id].documentos,
                    lambda p: mod_salida.construir_fragmentos(
                        ordenados[p.query_id].fragmentos)),
        juicios, empar, mapa)


def _evaluar_oraculo(pool, preguntas, encoders, prof, juicios, empar, mapa, resolver):
    docs, frags = {}, {}
    for p in preguntas:
        em = empar[p.query_id]
        relevantes = {resolver(d) for d in juicios[p.query_id].documentos}

        vistos, candidatos = set(), []
        for e in encoders:
            for meta, _ in pool[p.query_id][e][:prof]:
                if meta["chunk_id"] not in vistos:
                    vistos.add(meta["chunk_id"])
                    candidatos.append(meta)

        # Fragmentos: los más relevantes de verdad, respetando el tope por doc.
        por_doc: dict[str, int] = defaultdict(int)
        elegidos = []
        for m in sorted(candidatos, key=lambda m: -em.relevancia(m["texto"])):
            if por_doc[m["doc_id"]] >= config.MAX_FRAGMENTOS_POR_DOC:
                continue
            por_doc[m["doc_id"]] += 1
            elegidos.append({"text": mod_salida.recortar_a_limite(
                m["texto"], m.get("idioma") or "es", config.MAX_PALABRAS_FRAGMENTO)})
            if len(elegidos) == config.TOP_FRAGMENTOS:
                break
        frags[p.query_id] = elegidos

        # Documentos: los relevantes que estén en el pool, y el resto de relleno.
        en_pool: list[str] = []
        for m in candidatos:
            d = resolver(m["doc_id"])
            if d not in en_pool:
                en_pool.append(d)
        elegidos_doc = [d for d in en_pool if d in relevantes][: config.TOP_DOCUMENTOS]
        elegidos_doc += [d for d in en_pool if d not in relevantes][
            : config.TOP_DOCUMENTOS - len(elegidos_doc)]
        docs[p.query_id] = elegidos_doc

    return metricas.evaluar_textual(
        _resultados(preguntas, lambda p: docs[p.query_id], lambda p: frags[p.query_id]),
        juicios, empar, mapa)


if __name__ == "__main__":
    sys.exit(main())
