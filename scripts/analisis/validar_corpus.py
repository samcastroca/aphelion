"""Repite la comparación de recetas sobre el corpus completo, no la submuestra.

**Por qué hace falta.** El barrido decide sobre `data/submuestra.json`, que son
353 de 1.813 documentos elegidos para que quepan el ground truth entero y los
negativos difíciles. El presupuesto de fragmentos se agotó ahí, así que el
relleno estratificado quedó en cero y la submuestra tiene el 75% de sus
documentos tocados por el ground truth, contra el 14,6% del corpus real. Es
cinco veces más densa en material relevante.

Eso no invalida el barrido —para eso está, para iterar rápido— pero sí quiere
decir que sus métricas absolutas son optimistas y que el orden entre
configuraciones puede cambiar cuando cada consulta compita contra 64.484
fragmentos en vez de 29.005, la mayoría de ellos distractores nuevos.

Antes de cambiar lo que se entrega, la candidata tiene que ganar **aquí**.

Usa los índices de `entrega/base_vectorial`, que ya están construidos: no
codifica nada y termina en minutos.

Uso:
    uv run python scripts/analisis/validar_corpus.py
    uv run python scripts/analisis/validar_corpus.py --recetas entrega,dos-encoders
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas, recuperacion, salida as mod_salida
from aphelion.evaluacion import emparejamiento, metricas, recetas as mod_recetas
from aphelion.indice import vectores

RAIZ = Path(__file__).resolve().parents[2]

# Las que compiten por entregarse. Las que cambian la fragmentación quedan fuera:
# exigirían re-fragmentar y recodificar el corpus entero, que es lo que este
# guion existe para no hacer.
COMPARABLES = ("entrega", "dos-encoders", "convexa", "filtrado", "familias")


def recetas_medibles(
    elegidas: dict[str, mod_recetas.Receta],
    disponible,
) -> tuple[dict[str, mod_recetas.Receta], list[tuple[str, str]]]:
    """Separa las recetas que estos índices pueden medir de las que no.

    Dos motivos para no poder: la receta fragmenta de otra forma —sus vectores
    serían de otro chunking y compararla aquí mediría algo que no es— o pide un
    encoder que no está indexado en el corpus.

    Se devuelven las omitidas con su motivo en vez de abortar: una receta que no
    se puede medir no debe impedir medir las demás, que es a lo que se vino.
    """
    medibles: dict[str, mod_recetas.Receta] = {}
    omitidas: list[tuple[str, str]] = []

    for nombre, receta in elegidas.items():
        if (receta.chunk != config.CHUNK_PRESUPUESTO
                or receta.solape != config.CHUNK_SOLAPE):
            omitidas.append((
                nombre,
                f"fragmenta a {receta.chunk}/{receta.solape:.2f} y estos índices "
                f"son de {config.CHUNK_PRESUPUESTO}/{config.CHUNK_SOLAPE:.2f}: "
                "haría falta re-indexar el corpus",
            ))
            continue
        sin_indice = [e for e in receta.encoders if not disponible(e)]
        if sin_indice:
            omitidas.append((
                nombre, f"necesita {sin_indice}, que no está indexado en el corpus"))
            continue
        medibles[nombre] = receta

    return medibles, omitidas


def mapa_doc_a_fuente(base: Path, encoders: list[str]) -> dict[str, str]:
    """doc_id -> fuente, que es la clave con la que empareja el jurado."""
    mapa: dict[str, str] = {}
    for encoder in encoders:
        ruta = vectores.carpeta_encoder(encoder, base) / "metadata.jsonl"
        if not ruta.exists():
            continue
        with ruta.open(encoding="utf-8") as fh:
            for linea in fh:
                if linea.strip():
                    registro = json.loads(linea)
                    mapa[registro["doc_id"]] = registro.get("fuente", registro["doc_id"])
        break
    return mapa


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recetas", default=",".join(COMPARABLES))
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--profundidad", type=int, default=config.CANDIDATOS_POR_INDICE)
    args = ap.parse_args()

    nombres = mod_recetas.seleccionar(args.recetas)
    elegidas = {n: mod_recetas.RECETAS[n] for n in nombres}

    disponible = lambda e: (  # noqa: E731
        vectores.carpeta_encoder(e, args.base) / "index.faiss").exists()
    elegidas, omitidas = recetas_medibles(elegidas, disponible)
    for nombre, motivo in omitidas:
        print(f"  {nombre} se omite: {motivo}")
    if not elegidas:
        print("no queda ninguna receta medible con los índices que hay")
        return 1

    pedidos = sorted({e for r in elegidas.values() for e in r.encoders})
    indices = {e: vectores.cargar(e, args.base) for e in pedidos}
    print(f"corpus: {len(next(iter(indices.values()))):,} fragmentos indexados")

    preguntas = mod_consultas.cargar()
    juicios = metricas.cargar_juicios(config.GROUND_TRUTH)
    empar = emparejamiento.emparejadores(
        juicios, emparejamiento.cargar_textos(config.GROUND_TRUTH_TEXTOS))
    mapa = mapa_doc_a_fuente(args.base, pedidos)
    print(f"ground truth: {len(juicios)} consultas\n")

    # Una pasada de búsqueda por encoder, reutilizada por todas las recetas:
    # dos modelos large residentes a la vez no caben en esta clase de máquina.
    cache: dict[str, dict[str, list]] = {p.query_id: {} for p in preguntas}
    for nombre in pedidos:
        print(f"buscando con {nombre} ...", flush=True)
        buscador = recuperacion.Recuperador({nombre: indices[nombre]})
        for pregunta in preguntas:
            cache[pregunta.query_id].update(
                buscador.buscar(pregunta, candidatos_por_indice=args.profundidad))
        del buscador
        from aphelion.indice import encoders as mod_enc
        mod_enc.cargar.cache_clear()

    resultados: dict[str, dict] = {}
    for nombre, receta in elegidas.items():
        politica = receta.politica()
        rec = recuperacion.Recuperador(
            indices, k0=politica["k0"], boost=politica["boost"],
            max_por_doc=politica["max_por_doc"],
            umbral_relativo=politica["umbral"])
        salida_recetas = []
        for p in preguntas:
            rankings = {e: cache[p.query_id][e][: politica["candidatos"]]
                        for e in receta.encoders}
            r = rec.ordenar(rankings, p, fusion=politica["fusion"],
                            agregacion=politica["agregacion"])
            salida_recetas.append({
                "query_id": r.query_id,
                "documents": [{"rank": i, "doc_id": d}
                              for i, d in enumerate(r.documentos, start=1)],
                "fragments": mod_salida.construir_fragmentos(
                    r.fragmentos, subdividir_largos=politica["subdividir"]),
            })
        resultados[nombre] = metricas.evaluar_textual(
            salida_recetas, juicios, empar, mapa)

    lineas: list[str] = []

    def di(texto: str = "") -> None:
        print(texto)
        lineas.append(texto)

    di()
    di(f"corpus: {len(next(iter(indices.values()))):,} fragmentos, "
       f"{len(juicios)} consultas")
    di(f"{'receta':<13} {'NDCG@10':>9} {'F1@3':>8}   frente a la entrega")
    base_ev = resultados.get("entrega")
    qids = sorted(next(iter(resultados.values()))["por_consulta"])
    filas = []
    for nombre, ev in sorted(resultados.items(), key=lambda kv: -kv[1]["ndcg@10"]):
        linea = f"{nombre:<13} {ev['ndcg@10']:>9.4f} {ev['f1@3']:>8.4f}"
        fila = {"receta": nombre, "ndcg@10": ev["ndcg@10"], "f1@3": ev["f1@3"]}
        if base_ev is not None and ev is not base_ev:
            for m in ("ndcg@10", "f1@3"):
                dif, p = metricas.p_valor_permutacion(
                    [ev["por_consulta"][q][m] for q in qids],
                    [base_ev["por_consulta"][q][m] for q in qids])
                fila[f"dif_{m}"] = round(dif, 4)
                fila[f"p_{m}"] = round(p, 4)
            linea += (f"   NDCG {fila['dif_ndcg@10']:+.4f} (p={fila['p_ndcg@10']:.4f})"
                      f"  F1 {fila['dif_f1@3']:+.4f} (p={fila['p_f1@3']:.4f})")
        di(linea)
        filas.append(fila)

    if base_ev is not None and len(resultados) > 1:
        comparaciones = 2 * (len(resultados) - 1)
        di()
        di(f"  {comparaciones} comparaciones: umbral corregido "
           f"{0.05 / comparaciones:.4f}, no 0.05.")

    carpeta = config.PRUEBAS / "corpus-completo"
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    with (carpeta / "metricas.jsonl").open("w", encoding="utf-8") as fh:
        for fila in filas:
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"\n-> {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
