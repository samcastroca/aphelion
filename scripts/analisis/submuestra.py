"""Elige un subconjunto de documentos representativo para iterar rápido.

**El problema.** Codificar los 64.484 fragmentos del corpus cuesta horas por
encoder. Un barrido que pruebe cuatro tamaños de chunk por seis encoders son
veinticuatro corridas de eso, y no cabe en el tiempo que queda. Sobre un
subconjunto sí.

**Qué tiene que cumplir el subconjunto.** Tres cosas, en este orden:

1. **Contener todo lo que el ground truth toca.** Si un documento juzgado no
   está, sus juicios se convierten en ceros y la métrica miente hacia abajo; si
   un documento marcado relevante no está, el F1@3 de esa consulta tiene techo
   artificialmente bajo. Son 265 documentos y no son negociables.

   No negociables de verdad: no basta con avisar. Un aviso en medio de cien
   líneas de salida pasa inadvertido y el barrido acaba midiendo sobre un
   subconjunto incompleto, que es el fallo que no se ve. Así que un documento
   juzgado que no esté en el índice se recupera del texto extraído —que es de
   donde el barrido re-fragmenta, no del índice— y si no se puede recuperar, no
   se escribe la submuestra y el guion falla.

2. **Contener los negativos difíciles.** Un subconjunto de material relevante más
   ruido aleatorio es *más fácil* que el corpus real, y una configuración puede
   verse bien ahí solo porque no compite contra nada. Los negativos que importan
   son los que los encoders ya colocan alto sin ser relevantes: se toman de los
   rankings profundos cacheados, de las posiciones que quedaron por debajo del
   pool de anotación.

3. **Parecerse al corpus en composición.** El resto se completa con un muestreo
   estratificado por (fenómeno, formato) que respeta las proporciones del corpus,
   para que la mezcla de idiomas y de tipos de archivo no se distorsione.

**Los documentos van enteros.** No se recortan ni se les quitan fragmentos,
aunque eso mantenga dentro documentos de 1.960 fragmentos. Dos razones: el
barrido re-fragmenta, así que necesita el texto completo; y una de las cosas que
compara es la agregación a documento —max pooling frente a suma— cuyo resultado
depende justamente de la distribución de longitudes. Recortar los documentos
largos sesgaría esa comparación a favor de la suma.

Uso:
    uv run python scripts/analisis/submuestra.py
    uv run python scripts/analisis/submuestra.py --objetivo 20000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from aphelion import config
from aphelion.evaluacion import metricas
from aphelion.indice import vectores

CACHE_RANKINGS = config.TRABAJO / "rankings.jsonl"

# Fragmentos que se busca tener en el subconjunto. Es un objetivo, no un límite:
# los 265 documentos obligatorios ya aportan unos 24.000 y no se recortan, así
# que por debajo de esa cifra el objetivo se ignora y se avisa.
OBJETIVO = 28_000

SEMILLA = 20260801


def metadata_por_doc(base: Path) -> dict[str, dict]:
    """doc_id -> {fragmentos, formato, fenomeno, idioma, fuente}."""
    nombres = vectores.encoders_disponibles(base)
    if not nombres:
        raise FileNotFoundError(f"no hay índices en {base}")

    ruta = vectores.carpeta_encoder(nombres[0], base) / "metadata.jsonl"
    docs: dict[str, dict] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            r = json.loads(linea)
            d = docs.setdefault(
                r["doc_id"],
                {
                    "fragmentos": 0,
                    "formato": r.get("formato", ""),
                    "fenomeno": r.get("fenomeno"),
                    "idioma": r.get("idioma", ""),
                    "fuente": r.get("fuente", ""),
                },
            )
            d["fragmentos"] += 1
    return docs


def obligatorios(juicios: dict) -> set[str]:
    """Documentos que el ground truth toca, por juicio o por relevancia."""
    docs: set[str] = set()
    for juicio in juicios.values():
        docs.update(juicio.documentos)
        for chunk_id in juicio.fragmentos:
            docs.add(chunk_id.rsplit("-chunk-", 1)[0])
    return docs


def comprobar_cobertura(
    juicios: dict, elegidos: set[str]
) -> tuple[int, int, int, int]:
    """(juicios dentro, juicios totales, docs relevantes dentro, totales).

    Se cuenta sobre los juicios y no sobre los documentos porque es la unidad en
    la que la métrica miente: un documento juzgado que falte convierte en ceros
    todos sus juicios, y son ellos los que entran en el NDCG@10.
    """
    juzgados = sum(len(j.fragmentos) for j in juicios.values())
    cubiertos = sum(
        1
        for j in juicios.values()
        for c in j.fragmentos
        if c.rsplit("-chunk-", 1)[0] in elegidos
    )
    relevantes = {d for j in juicios.values() for d in j.documentos}
    return cubiertos, juzgados, len(relevantes & elegidos), len(relevantes)


def desde_texto_crudo(doc_id: str, raiz: Path) -> dict | None:
    """Metadata de un documento juzgado que no está en el índice.

    Estar fuera del índice no lo deja fuera del alcance: lo que el barrido
    fragmenta es `trabajo/texto/<doc_id>.json` —`03_fragmentar --docs` filtra esa
    carpeta por el nombre del archivo—, así que un documento extraído se puede
    incluir aunque el índice de referencia no lo tenga. Pasa cuando el índice se
    construyó antes de que el documento se extrajera, o cuando se anotó sobre un
    índice y se muestrea contra otro.

    Devuelve None si tampoco hay texto extraído: entonces el documento no es
    alcanzable por ninguna vía y quien llama tiene que parar.

    `fragmentos` queda en 0 porque nadie lo ha fragmentado todavía. Solo se usa
    para presupuestar el relleno, y contar de menos hace que el subconjunto
    quede algo más grande que el objetivo, nunca que se caiga un obligatorio.
    """
    ruta = raiz / f"{doc_id}.json"
    if not ruta.exists():
        return None

    doc = json.loads(ruta.read_text(encoding="utf-8"))
    fuente = doc.get("fuente", "")
    return {
        "fragmentos": 0,
        # Mismo criterio que `chunking.fragmentar_documento`: el tipo declarado
        # y, si falta, la extensión del nombre del archivo.
        "formato": (doc.get("tipo") or fuente.rsplit(".", 1)[-1]).lower(),
        "fenomeno": doc.get("fenomeno"),
        "idioma": "",  # lo decide la fragmentación, que aún no ha corrido
        "fuente": fuente,
    }


def negativos_difciles(ruta: Path, ya: set[str]) -> list[str]:
    """Documentos que los encoders puntúan alto sin haber sido juzgados.

    Salen de los rankings profundos que `comparar_encoders.py` cacheó. El orden
    de preferencia es la mejor posición que el documento alcanzó en cualquier
    encoder y consulta: cuanto más arriba llegó sin ser relevante, más
    discriminativo es como distractor.
    """
    if not ruta.exists():
        print(f"  aviso: no existe {ruta}; sin negativos difíciles.")
        print("  Genéralos con scripts/analisis/comparar_encoders.py")
        return []

    mejor: dict[str, int] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            registro = json.loads(linea)
            for ranking in registro["rankings"].values():
                for posicion, entrada in enumerate(ranking, start=1):
                    doc = entrada["chunk_id"].rsplit("-chunk-", 1)[0]
                    if doc in ya:
                        continue
                    if posicion < mejor.get(doc, 10**9):
                        mejor[doc] = posicion

    return [doc for doc, _ in sorted(mejor.items(), key=lambda kv: kv[1])]


def estratos(docs: dict[str, dict]) -> dict[tuple, list[str]]:
    por_estrato: dict[tuple, list[str]] = defaultdict(list)
    for doc_id, d in docs.items():
        por_estrato[(d["fenomeno"], d["formato"])].append(doc_id)
    return por_estrato


def resumen(etiqueta: str, elegidos: set[str], docs: dict[str, dict]) -> None:
    frag = sum(docs[d]["fragmentos"] for d in elegidos)
    print(f"\n{etiqueta}: {len(elegidos):,} documentos, {frag:,} fragmentos")
    for clave in ("formato", "fenomeno", "idioma"):
        cuenta = Counter(docs[d][clave] for d in elegidos)
        print(f"  por {clave:<9} {dict(sorted(cuenta.items(), key=lambda kv: str(kv[0])))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=config.BASE_VECTORIAL)
    ap.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH)
    ap.add_argument("--salida", type=Path, default=config.SUBMUESTRA)
    ap.add_argument("--objetivo", type=int, default=OBJETIVO,
                    help="fragmentos que se busca tener en total")
    ap.add_argument("--negativos", type=int, default=250,
                    help="cuántos negativos difíciles añadir como máximo")
    args = ap.parse_args()

    if not args.ground_truth.exists():
        print(f"no existe {args.ground_truth}. Anota el ground truth primero.")
        return 1

    juicios = metricas.cargar_juicios(args.ground_truth)
    docs = metadata_por_doc(args.base)
    indexados = len(docs)
    total_frag = sum(d["fragmentos"] for d in docs.values())
    print(f"corpus indexado: {indexados:,} documentos, {total_frag:,} fragmentos")

    # 1. Lo que el ground truth toca. Va entero y sin discusión: lo que no esté
    # en el índice se busca en el texto extraído, que es de donde el barrido
    # fragmenta, y lo que no aparezca por ninguna de las dos vías detiene la
    # corrida. Un obligatorio que se cae no da un error, da una métrica más baja
    # —sus juicios se vuelven ceros— y eso no se distingue de una configuración
    # peor.
    nucleo = obligatorios(juicios)
    recuperados: list[str] = []
    irrecuperables: list[str] = []
    for doc_id in sorted(nucleo - set(docs)):
        registro = desde_texto_crudo(doc_id, config.TEXTO_CRUDO)
        if registro is None:
            irrecuperables.append(doc_id)
        else:
            docs[doc_id] = registro
            recuperados.append(doc_id)

    if recuperados:
        print(f"\n  {len(recuperados)} documentos juzgados no están en el índice y se "
              f"toman del texto extraído:")
        print(f"    {', '.join(recuperados[:5])}"
              f"{' ...' if len(recuperados) > 5 else ''}")

    if irrecuperables:
        print(f"\nERROR: {len(irrecuperables)} documentos del ground truth no están "
              f"ni en el índice ni en {config.TEXTO_CRUDO}:")
        for doc_id in irrecuperables[:10]:
            print(f"    {doc_id}")
        if len(irrecuperables) > 10:
            print(f"    ... y {len(irrecuperables) - 10} más")
        print("\nSin ellos la submuestra mediría hacia abajo sin decirlo, así que no")
        print("se escribe. Extrae esos documentos (scripts/etapas/01_extraer.py) o")
        print("corrige el ground truth, y repite.")
        return 1

    elegidos = set(nucleo)
    resumen("núcleo del ground truth", elegidos, docs)

    frag = sum(docs[d]["fragmentos"] for d in elegidos)
    print(f"\n  ese núcleo es el suelo del subconjunto: {frag / total_frag:.0%} de los")
    print("  fragmentos del corpus. Los documentos van enteros y no se recortan,")
    print("  por dos razones: el barrido re-fragmenta y necesita el texto completo,")
    print("  y una de las cosas que compara es la agregación a documento —max")
    print("  frente a suma— cuyo resultado depende de la longitud real. Recortar")
    print("  los documentos largos sesgaría esa comparación a favor de la suma.")
    if frag >= args.objetivo:
        print(f"\n  el objetivo de {args.objetivo:,} queda por debajo del suelo y se ignora.")

    # 2. Negativos difíciles: lo que los encoders suben sin ser relevante. Se
    # añaden en orden de mejor posición alcanzada y **hasta agotar el objetivo**,
    # no a cuenta libre: son documentos que compiten, y por eso mismo suelen ser
    # largos. Sin este tope 250 de ellos añaden diez mil fragmentos y se comen
    # la ventaja de trabajar con una submuestra.
    # La caché de rankings puede ser de un índice anterior: un doc_id que ya no
    # exista pasaría de largo hasta reventar contra `docs[doc]`, y el fallo sería
    # tumbar la submuestra entera por un negativo prescindible.
    candidatos_duros = [
        d for d in negativos_difciles(CACHE_RANKINGS, elegidos) if d in docs
    ]
    duros: list[str] = []
    for doc in candidatos_duros:
        if len(duros) >= args.negativos or frag >= args.objetivo:
            break
        duros.append(doc)
        elegidos.add(doc)
        frag += docs[doc]["fragmentos"]
    if duros:
        print(f"\nnegativos difíciles añadidos: {len(duros)} de "
              f"{len(candidatos_duros)} candidatos")
        if len(duros) < min(args.negativos, len(candidatos_duros)):
            print("  se cortó al alcanzar el objetivo de fragmentos")

    # 3. Relleno estratificado hasta el objetivo, respetando las proporciones del
    # corpus por (fenómeno, formato).
    rng = random.Random(SEMILLA)
    disponibles = estratos({d: v for d, v in docs.items() if d not in elegidos})
    pesos = {clave: len(v) for clave, v in disponibles.items()}
    total_pesos = sum(pesos.values()) or 1
    for lista in disponibles.values():
        rng.shuffle(lista)

    añadidos = 0
    # Ronda a ronda, tomando de cada estrato en proporción a su tamaño. Así el
    # relleno converge a la composición del corpus en vez de agotar un estrato.
    while frag < args.objetivo and any(disponibles.values()):
        for clave, lista in sorted(disponibles.items(), key=lambda kv: str(kv[0])):
            if not lista or frag >= args.objetivo:
                continue
            # cuántos de este estrato por ronda, mínimo uno
            cuota = max(1, round(10 * pesos[clave] / total_pesos))
            for _ in range(cuota):
                if not lista or frag >= args.objetivo:
                    break
                doc = lista.pop()
                elegidos.add(doc)
                frag += docs[doc]["fragmentos"]
                añadidos += 1
    print(f"\nrelleno estratificado: {añadidos} documentos")

    resumen("SUBMUESTRA", elegidos, docs)
    frag = sum(docs[d]["fragmentos"] for d in elegidos)
    print(f"\n  {len(elegidos) / indexados:.1%} de los documentos, "
          f"{frag / total_frag:.1%} de los fragmentos")
    print(f"  la codificación debería tardar ~{total_frag / max(frag, 1):.1f}x menos")

    # Cobertura del ground truth: es el invariante que da sentido al subconjunto,
    # así que se comprueba y decide, no se informa. Tras el paso 1 tiene que salir
    # total por construcción; que aquí falle significa que algo la deshizo después,
    # y entonces lo que no puede pasar es que el archivo se escriba igual.
    cubiertos, juzgados, dentro_rel, total_rel = comprobar_cobertura(juicios, elegidos)
    print(f"\ncobertura del ground truth:")
    print(f"  juicios de fragmento dentro: {cubiertos}/{juzgados}")
    print(f"  documentos relevantes dentro: {dentro_rel}/{total_rel}")
    if cubiertos < juzgados or dentro_rel < total_rel:
        print("\nERROR: la submuestra no cubre el ground truth entero. No se escribe:")
        print("  medir sobre ella daría números más bajos sin que nada lo indique.")
        return 1

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(
            {
                "doc_ids": sorted(elegidos),
                "documentos": len(elegidos),
                "fragmentos_en_la_fragmentacion_de_referencia": frag,
                "objetivo": args.objetivo,
                "nucleo_ground_truth": len(nucleo),
                "negativos_dificiles": len(duros),
                "relleno_estratificado": añadidos,
                "semilla": SEMILLA,
                "ground_truth": str(args.ground_truth),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> {args.salida}")
    print("\nSiguiente paso: uv run python scripts/analisis/barrido_completo.py --rapido")
    return 0


if __name__ == "__main__":
    sys.exit(main())
