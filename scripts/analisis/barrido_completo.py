"""Corre un experimento: las opciones que se le pidan, y guarda sus métricas.

**No barre todo.** Cada dimensión —encoders, tamaño de chunk, solape, fusión,
agregación, realce, diversificación, profundidad, umbral— se pide por separado y
se prueba solo lo elegido. El producto completo son decenas de miles de corridas
y varias noches de GPU; elegir tres cosas a la vez y mirar el resultado es más
rápido de interpretar y no sobreajusta tanto.

**Tres formas de decirle qué probar**, de la más concreta a la más amplia:

- `--recetas` corre **configuraciones completas** que ya traen todos los
  parámetros fijados, cada una con su apuesta y su motivo
  (`aphelion.evaluacion.recetas`). Una receta es una corrida, así que la tabla
  se lee sin descontar sobreajuste. Es lo que se usa para decidir qué entregar.
- `--preset` abre **una** dimensión y deja las demás quietas, para entender qué
  efecto tiene esa dimensión y no para elegir campeón.
- las opciones sueltas (`--encoders`, `--chunks`, ...) construyen la rejilla a
  mano, y lo que no se diga toma su valor de la entrega.

**Dónde queda cada cosa.**

    pruebas/fragmentos/<chunking>.jsonl        compartido entre experimentos
    pruebas/indices/<chunking>/encoder_<x>/    compartido entre experimentos
    pruebas/<nombre>/metricas.jsonl            una línea por corrida
    pruebas/<nombre>/resumen.txt               la tabla que se imprime
    pruebas/<nombre>/config.json              qué se pidió, para repetirlo

Los fragmentos y los índices se comparten porque son caros y su contenido queda
determinado por (chunking, encoder): dos experimentos que pidan el mismo índice
lo reutilizan en vez de duplicar 117 MB. Las métricas van por experimento, que es
lo que se compara entre ellos.

**Lo caro y lo barato.** Cambiar encoder, chunk o solape obliga a re-fragmentar y
recodificar, y se cachea. Cambiar fusión, k₀, realce, diversificación, umbral o
agregación es reordenar candidatos que ya están en memoria: se busca una vez por
consulta y se reordena tantas veces como políticas haya.

**Emparejamiento por texto.** El ground truth se anotó sobre chunks de 504
tokens; con otro tamaño sus `chunk_id` no existen. La relevancia se decide por
solape de texto, que además es como lo hará el jurado (§10.2.1).

**Sin modelos generativos**: todos los encoders del catálogo son arquitecturas
encoder con licencia libre, y fusión, filtrado y agregación operan sobre
vectores, puntuaciones y metadata (§4.2, §8.3).

Uso:
    # Candidatas completas, cada una con todos sus parámetros ya fijados
    uv run python -m aphelion.evaluacion.recetas             # ver el catálogo
    uv run python scripts/analisis/barrido_completo.py --recetas entrega,dos-encoders
    uv run python scripts/analisis/barrido_completo.py --recetas todas

    # Preguntando nada: los dos encoders de la entrega, chunking actual
    uv run python scripts/analisis/barrido_completo.py --nombre base

    # Elegir a mano cada dimensión
    uv run python scripts/analisis/barrido_completo.py --nombre mi-prueba \\
        --encoders bge-m3,gte-multilingual-base --chunks 504 --solapes 0.15 \\
        --fusiones rrf,convexa --agregaciones max,top2

    # Preguntas frecuentes, ya preparadas
    uv run python scripts/analisis/barrido_completo.py --preset chunking
    uv run python scripts/analisis/barrido_completo.py --preset fusion
    uv run python scripts/analisis/barrido_completo.py --listar
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from aphelion import config
from aphelion.busqueda import consultas as mod_consultas, recuperacion, salida
from aphelion.evaluacion import emparejamiento, metricas, recetas as mod_recetas
from aphelion.indice import vectores

RAIZ = Path(__file__).resolve().parents[2]
ETAPAS = RAIZ / "scripts" / "etapas"

PROFUNDIDAD_MAXIMA = 200

# Lo que se prueba si no se pide otra cosa: los dos encoders de la entrega sobre
# el chunking de la entrega, variando solo lo que es gratis variar.
# Los encoders por defecto son los dos grandes y no `config.ENCODERS_ENTREGA`:
# desde que la entrega es de un solo encoder, atarlos dejaría al barrido sin
# nada que fusionar, y comparar fusiones es la mitad de para lo que existe.
DEFECTOS: dict[str, list] = {
    "encoders": ["bge-m3", "me5-large"],
    "chunks": [config.CHUNK_PRESUPUESTO],
    "solapes": [config.CHUNK_SOLAPE],
    "fusiones": ["rrf", "convexa"],
    "k0": [config.RRF_K0],
    "boosts": [1.0, config.BOOST_FENOMENO],
    "max_por_doc": [config.MAX_FRAGMENTOS_POR_DOC],
    "candidatos": [config.CANDIDATOS_POR_INDICE],
    "umbrales": [None],
    "agregaciones": ["max", "top2"],
    "subdividir": [False],
}

# Selecciones ya hechas para las preguntas que suelen interesar. Cada una abre
# una dimensión y deja las demás fijas, que es como se lee un resultado sin
# confundir el efecto de una cosa con el de otra.
PRESETS: dict[str, dict] = {
    # Valida la maquinaria en minutos con el encoder más barato. No decide nada.
    "rapido": {
        "encoders": ["me5-small"],
        "fusiones": ["rrf"],
        "boosts": [1.0],
        "agregaciones": ["max"],
    },
    # ¿Qué tamaño de chunk y cuánto solape? Con el encoder más barato, porque hay
    # que recodificar seis veces y el orden entre chunkings no depende del modelo.
    #
    # 768 no está y no es un olvido: la ventana de mE5 es de 512, así que ese caso
    # se codificaría truncado y se leería como si hubiera medido el chunking largo.
    # Para probar 768 hace falta un encoder de ventana larga: `--chunks 768
    # --encoders bge-m3` o la receta `contexto`. En su lugar entra 345, que es el
    # presupuesto con el que el 90% de los fragmentos cabe en las 250 palabras que
    # exige el reto sin recortarse.
    "chunking": {
        "encoders": ["me5-small"],
        "chunks": [256, 345, 504],
        "solapes": [0.0, 0.15],
        "fusiones": ["rrf"],
        "boosts": [1.0],
        "agregaciones": ["max"],
    },
    # ¿Qué encoder solo, y qué pares? Sobre el chunking que haya ganado.
    "encoders": {
        "encoders": ["bge-m3", "me5-large", "me5-base", "gte-multilingual-base"],
        "fusiones": ["rrf", "convexa"],
        "agregaciones": ["max", "top2"],
    },
    # ¿Cómo se combinan dos espacios vectoriales? Es la pregunta que el diseño
    # dejó abierta: RRF frente a CombSUM frente a la convexa normalizada.
    "fusion": {
        "fusiones": ["rrf", "combsum", "combmnz", "convexa"],
        "k0": [10, 20, 60],
        "boosts": [1.0],
        "agregaciones": ["max"],
    },
    # ¿Cómo se pasa de fragmentos a documentos? Es donde la fusión gana hoy.
    "agregacion": {
        "fusiones": ["rrf"],
        "boosts": [1.0],
        "agregaciones": ["max", "suma", "media", "top2", "top3"],
        "max_por_doc": [2, 3, 5],
    },
    # Todo lo barato, con el chunking y los encoders fijos.
    "politicas": {
        "fusiones": ["rrf", "combsum", "combmnz", "convexa"],
        "k0": [10, 20, 60],
        "boosts": [1.0, 1.05, 1.15],
        "max_por_doc": [2, 3, 5],
        "candidatos": [100, 200],
        "umbrales": [None, 0.9],
        "agregaciones": ["max", "suma", "top2", "top3"],
        "subdividir": [False, True],
    },
    # El catálogo entero, incluido el control. Es una noche de GPU.
    #
    # Aquí sí conviven chunkings con encoders que no los admiten, y el guion los
    # descarta avisando en vez de indexarlos truncados. Dos casos: 768 excluye a
    # toda la familia mE5 (ventana 512), y LaBSE —ventana 256— no cabe en ninguno
    # de estos presupuestos, así que medir el control exige su propia corrida:
    # `--encoders labse --chunks 248`.
    "todo": {
        "encoders": sorted(config.ENCODERS),
        "chunks": [256, 504, 768],
        "solapes": [0.0, 0.15],
        "fusiones": ["rrf", "combsum", "combmnz", "convexa"],
        "agregaciones": ["max", "top2", "top3"],
    },
}


# --- Rutas ----------------------------------------------------------------

# Viven en config porque las comparten quien escribe los artefactos, quien
# decide si ya existen y el catálogo de recetas, que enseña cuántos índices le
# faltan a cada una. Dos versiones de esta cadena harían que la caché no
# acertara y se reindexara lo que ya estaba.
clave_chunking = config.clave_chunking
ruta_fragmentos = config.ruta_fragmentos_prueba
ruta_indices = config.ruta_indices_prueba


# --- Qué se va a correr ---------------------------------------------------


@dataclass
class Plan:
    """Una corrida por medir: qué índices usa y con qué política los ordena.

    La rejilla y las recetas producen lo mismo —una lista de planes— y de ahí
    para abajo el guion no distingue de dónde salieron. Esa es la razón de que
    exista: sin ella, correr configuraciones concretas obligaría a expresarlas
    como un producto cartesiano de un solo elemento por dimensión, y eso no se
    puede hacer cuando dos recetas piden chunkings distintos con encoders
    distintos.
    """

    encoders: tuple[str, ...]
    chunk: int
    solape: float
    cfg: dict
    receta: str | None = None


# --- Rejilla --------------------------------------------------------------


def politicas(op: dict) -> list[dict]:
    """El producto de las dimensiones baratas, sin duplicados.

    `k0` solo interviene en RRF: con CombSUM o la convexa, variarlo produce
    corridas idénticas que multiplicarían el tiempo y falsearían el Conteo de
    Borda al premiar una configuración por aparecer repetida.
    """
    claves = ["fusion", "k0", "boost", "max_por_doc", "candidatos", "umbral",
              "subdividir", "agregacion"]
    valores = [op["fusiones"], op["k0"], op["boosts"], op["max_por_doc"],
               op["candidatos"], op["umbrales"], op["subdividir"], op["agregaciones"]]

    vistos, unicos = set(), []
    for combo in itertools.product(*valores):
        cfg = dict(zip(claves, combo))
        firma = tuple(
            sorted((k, v) for k, v in cfg.items() if not (k == "k0" and cfg["fusion"] != "rrf"))
        )
        if firma in vistos:
            continue
        vistos.add(firma)
        unicos.append(cfg)
    return unicos


def subconjuntos(encoders: list[str], maximo: int) -> list[tuple[str, ...]]:
    """Los encoders solos y sus combinaciones hasta `maximo`.

    Solos porque el hallazgo actual es que el mejor individual gana en
    fragmentos; combinados porque gana en documentos. Las dos cosas hay que
    medirlas y ninguna se puede dar por hecha.
    """
    salida: list[tuple[str, ...]] = []
    for n in range(1, min(maximo, len(encoders)) + 1):
        salida.extend(itertools.combinations(encoders, n))
    return salida


def planificar_rejilla(op: dict) -> list[Plan]:
    """El producto de lo que se pidió: chunkings x encoders x políticas."""
    politicas_ = politicas(op)
    planes = []
    for chunk, solape in itertools.product(op["chunks"], op["solapes"]):
        for encs in subconjuntos(op["encoders"], op["max_fusion"]):
            planes.extend(Plan(encs, chunk, solape, cfg) for cfg in politicas_)
    return planes


def planificar_recetas(nombres: list[str]) -> list[Plan]:
    """Una corrida por receta, con sus parámetros tal como los declara."""
    return [
        Plan(
            encoders=mod_recetas.RECETAS[n].encoders,
            chunk=mod_recetas.RECETAS[n].chunk,
            solape=mod_recetas.RECETAS[n].solape,
            cfg=mod_recetas.RECETAS[n].politica(),
            receta=n,
        )
        for n in nombres
    ]


def avisar_de_ventanas(planes: list[Plan]) -> None:
    """Avisa de los pares (chunk, encoder) donde el fragmento no cabe.

    No aborta: un barrido que incluya un chunking grande y un encoder de ventana
    corta sigue siendo informativo para los otros pares. Lo que no puede pasar es
    que la corrida truncada se lea como si hubiera medido el chunking entero.
    """
    malos = sorted(
        {
            (p.chunk, e)
            for p in planes
            for e in p.encoders
            if not config.cabe_en_ventana(p.chunk, e)
        }
    )
    for chunk, encoder in malos:
        print(
            f"  ! {chunk} tokens no caben en {encoder} "
            f"(ventana {config.ENCODERS[encoder]['max_tokens']}): sus vectores "
            "no representarán la cola del fragmento"
        )


@dataclass
class Corrida:
    encoders: tuple[str, ...]
    chunk: int
    solape: float
    cfg: dict
    ndcg: float
    f1: float
    ic_ndcg: tuple[float, float]
    ic_f1: tuple[float, float]
    receta: str | None = None
    # query_id -> {'ndcg@10', 'f1@3'}. Hace falta para comparar dos corridas
    # consulta a consulta; las medias solas no permiten parear.
    por_consulta: dict[str, dict] = field(default_factory=dict)

    @property
    def etiqueta(self) -> str:
        enc = "+".join(e.replace("multilingual-", "") for e in self.encoders)
        c = self.cfg
        k0 = f"k0={c['k0']:<3}" if c["fusion"] == "rrf" else "       "
        return (
            f"{enc:<28} {self.chunk}/{self.solape:.2f} {c['fusion']:<8} {k0} "
            f"b={c['boost']:<5} d={c['max_por_doc']} n={c['candidatos']:<4} "
            f"u={('no' if not c['umbral'] else str(c['umbral'])):<4} "
            f"s={'sí' if c['subdividir'] else 'no'} a={c['agregacion']:<5}"
        )

    def to_dict(self, borda: int, p: dict[str, float] | None = None) -> dict:
        # `por_consulta` se queda fuera a propósito: son cincuenta entradas por
        # corrida y lo que se compara entre experimentos son los agregados.
        return {
            "receta": self.receta,
            "encoders": list(self.encoders), "chunk": self.chunk,
            "solape": self.solape, **self.cfg,
            "ndcg@10": self.ndcg, "f1@3": self.f1,
            "ic_ndcg@10": list(self.ic_ndcg), "ic_f1@3": list(self.ic_f1),
            **(p or {}),
            "borda": borda,
        }


def comparar_con_la_base(corrida: Corrida, base: Corrida) -> dict[str, float]:
    """p-valores pareados de una corrida frente a la línea base.

    Devuelve `{}` si a alguna de las dos le faltan las métricas por consulta,
    que es lo que pasa al releer un experimento antiguo: sin ellas no se puede
    parear, y un p-valor calculado sobre otra cosa sería peor que ninguno.
    """
    comunes = sorted(set(corrida.por_consulta) & set(base.por_consulta))
    if not comunes:
        return {}
    salida: dict[str, float] = {}
    for metrica in ("ndcg@10", "f1@3"):
        _, p = metricas.p_valor_permutacion(
            [corrida.por_consulta[q][metrica] for q in comunes],
            [base.por_consulta[q][metrica] for q in comunes],
        )
        salida[f"p_{metrica}"] = round(p, 4)
    return salida


# --- Fase cara ------------------------------------------------------------


def correr(comando: list[str], etiqueta: str) -> None:
    print(f"  -> {etiqueta}", flush=True)
    codigo = subprocess.call([sys.executable, *comando], cwd=RAIZ)
    if codigo != 0:
        raise SystemExit(f"{etiqueta} falló (código {codigo})")


def pares_a_indexar(planes: list[Plan]) -> dict[tuple[int, float], list[str]]:
    """Qué encoder hay que indexar en qué fragmentación.

    Se saca de los planes y no del producto de las opciones porque una receta
    empareja un chunking con unos encoders concretos: `contexto` pide 768 tokens
    con BGE-M3 y nada más, y codificar ahí mE5 sería gastar una hora de GPU en un
    índice truncado que ningún plan va a consultar.

    Los pares donde el fragmento no cabe en la ventana del encoder se descartan:
    su índice representaría fragmentos truncados y daría métricas plausibles de un
    chunking que nadie midió. Las corridas que dependan de ellos se omiten después,
    por el mismo camino que las que les falta un índice.
    """
    pares: dict[tuple[int, float], list[str]] = {}
    for plan in planes:
        cabe = [e for e in plan.encoders if config.cabe_en_ventana(plan.chunk, e)]
        if not cabe:
            continue
        faltan = pares.setdefault((plan.chunk, plan.solape), [])
        faltan.extend(e for e in cabe if e not in faltan)
    return pares


def preparar(planes: list[Plan], submuestra: Path, backend: str, lote: int | None) -> None:
    """Fragmenta e indexa lo que falte. Lo que ya esté se reutiliza."""
    for (chunk, solape), encoders in pares_a_indexar(planes).items():
        clave = clave_chunking(chunk, solape)
        frag = ruta_fragmentos(chunk, solape)
        if frag.exists() and frag.stat().st_size > 0:
            print(f"{clave}: fragmentos reutilizados")
        else:
            frag.parent.mkdir(parents=True, exist_ok=True)
            correr(
                [str(ETAPAS / "03_fragmentar.py"),
                 "--max-tokens", str(chunk), "--solape", str(solape),
                 "--salida", str(frag), "--docs", str(submuestra)],
                f"fragmentar {clave}",
            )

        base = ruta_indices(chunk, solape)
        for encoder in encoders:
            if (base / f"encoder_{encoder}" / "index.faiss").exists():
                print(f"  {encoder}: índice reutilizado")
                continue
            comando = [
                str(ETAPAS / "04_indexar.py"),
                "--encoder", encoder, "--fragmentos", str(frag),
                "--base", str(base), "--backend", backend,
            ]
            if lote:
                comando += ["--lote", str(lote)]
            correr(comando, f"indexar {encoder} en {clave}")


# --- Fase barata ----------------------------------------------------------


def liberar_encoders() -> None:
    """Suelta los modelos cargados. Igual que hace 06_verificar antes del entregable."""
    from aphelion.indice import encoders as mod_encoders

    mod_encoders.cargar.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def mapa_doc_a_fuente(base: Path) -> dict[str, str]:
    nombres = vectores.encoders_disponibles(base)
    ruta = vectores.carpeta_encoder(nombres[0], base) / "metadata.jsonl"
    mapa: dict[str, str] = {}
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                r = json.loads(linea)
                mapa.setdefault(r["doc_id"], r["fuente"])
    return mapa


def evaluar_chunking(
    chunk: int, solape: float, planes: list[Plan], juicios: dict, empar: dict,
    preguntas: list,
) -> list[Corrida]:
    """Mide los planes de una fragmentación, buscando una sola vez por consulta."""
    clave = clave_chunking(chunk, solape)
    base = ruta_indices(chunk, solape)

    pedidos: list[str] = []
    for plan in planes:
        pedidos.extend(e for e in plan.encoders if e not in pedidos)

    # Aunque exista un índice de una corrida anterior, si el fragmento no cabe en
    # la ventana del encoder ese índice está truncado y no se usa.
    no_caben = [e for e in pedidos if not config.cabe_en_ventana(chunk, e)]
    if no_caben:
        print(f"  {clave}: {no_caben} no admiten {chunk} tokens, se descartan")
    hay = [
        e for e in pedidos
        if e not in no_caben and (base / f"encoder_{e}" / "index.faiss").exists()
    ]
    if not hay:
        print(f"  {clave}: sin índices utilizables, se omite")
        return []

    ejecutables = [p for p in planes if all(e in hay for e in p.encoders)]
    if len(ejecutables) < len(planes):
        faltan = sorted(set(pedidos) - set(hay))
        print(f"  {clave}: sin índice de {faltan}; "
              f"se omiten {len(planes) - len(ejecutables)} corridas")
    if not ejecutables:
        return []

    indices = recuperacion.cargar_indices(hay, base)
    mapa = mapa_doc_a_fuente(base)
    profundidad = min(PROFUNDIDAD_MAXIMA, max(p.cfg["candidatos"] for p in ejecutables))

    # Un encoder a la vez, y se suelta al acabar su pasada. Dos modelos large
    # residentes no caben en una máquina de esta clase: cargar BGE-M3 y mE5-large
    # juntos para codificar cincuenta consultas revienta con «el archivo de
    # paginación es demasiado pequeño» (os error 1455). Como después solo se
    # reordena lo cacheado, ninguno hace falta más allá de su pasada.
    cache: dict[str, dict[str, list]] = {p.query_id: {} for p in preguntas}
    for nombre in hay:
        print(f"  buscando con {nombre}: {len(preguntas)} consultas ...", flush=True)
        buscador = recuperacion.Recuperador({nombre: indices[nombre]})
        for pregunta in tqdm(preguntas, desc="    consultas", unit="q", leave=False):
            cache[pregunta.query_id].update(
                buscador.buscar(pregunta, candidatos_por_indice=profundidad)
            )
        del buscador
        liberar_encoders()
    print(f"  {len(ejecutables)} corridas sobre esos rankings")

    corridas: list[Corrida] = []
    for plan in tqdm(ejecutables, desc="    corridas", unit="cfg", leave=False):
        cfg = plan.cfg
        recuperador = recuperacion.Recuperador(
            indices, k0=cfg["k0"], boost=cfg["boost"],
            max_por_doc=cfg["max_por_doc"], umbral_relativo=cfg["umbral"],
        )
        resultados = []
        for pregunta in preguntas:
            rankings = {
                n: lista[: cfg["candidatos"]]
                for n, lista in cache[pregunta.query_id].items()
                if n in plan.encoders
            }
            r = recuperador.ordenar(
                rankings, pregunta,
                fusion=cfg["fusion"], agregacion=cfg["agregacion"],
            )
            resultados.append({
                "query_id": r.query_id,
                "documents": [
                    {"rank": i, "doc_id": d}
                    for i, d in enumerate(r.documentos, start=1)
                ],
                "fragments": salida.construir_fragmentos(
                    r.fragmentos, subdividir_largos=cfg["subdividir"]
                ),
            })

        resumen = metricas.evaluar_textual(resultados, juicios, empar, mapa)
        corridas.append(Corrida(
            encoders=plan.encoders, chunk=chunk, solape=solape, cfg=cfg,
            ndcg=resumen["ndcg@10"], f1=resumen["f1@3"],
            ic_ndcg=resumen["ic_ndcg@10"], ic_f1=resumen["ic_f1@3"],
            receta=plan.receta, por_consulta=resumen["por_consulta"],
        ))
    return corridas


def puntos_borda(corridas: list[Corrida]) -> list[int]:
    n = len(corridas)
    puntos = [0] * n
    for clave in (lambda c: c.ndcg, lambda c: c.f1):
        orden = sorted(range(n), key=lambda i: clave(corridas[i]), reverse=True)
        for posicion, i in enumerate(orden, start=1):
            puntos[i] += n - posicion
    return puntos


def informar(corridas: list[Corrida], top: int, carpeta: Path) -> None:
    lineas: list[str] = []

    def di(texto: str = "") -> None:
        print(texto)
        lineas.append(texto)

    if not corridas:
        di("ninguna corrida evaluable")
        return

    puntos = puntos_borda(corridas)
    orden = sorted(range(len(corridas)), key=lambda i: puntos[i], reverse=True)

    # La columna de receta solo aparece si se corrieron recetas: en una rejilla
    # estaría vacía en todas las filas y solo quitaría sitio a la configuración.
    hay_recetas = any(c.receta for c in corridas)
    nombre_de = (lambda c: f"{(c.receta or ''):<13} ") if hay_recetas else (lambda c: "")

    columna_receta = f"{'receta':<13} " if hay_recetas else ""
    di("")
    di(f"{'':<4} {columna_receta}{'configuración':<106} "
       f"{'NDCG@10':>8} {'F1@3':>8} {'Borda':>7}")
    for posicion, i in enumerate(orden[:top], start=1):
        c = corridas[i]
        di(f"{posicion:>4} {nombre_de(c)}{c.etiqueta:<106} "
           f"{c.ndcg:>8.4f} {c.f1:>8.4f} {puntos[i]:>7}")

    mejor_n = max(corridas, key=lambda c: c.ndcg)
    mejor_f = max(corridas, key=lambda c: c.f1)
    di("")
    di(f"mejor NDCG@10: {mejor_n.ndcg:.4f}  IC [{mejor_n.ic_ndcg[0]:.3f}, {mejor_n.ic_ndcg[1]:.3f}]")
    di(f"  {nombre_de(mejor_n)}{mejor_n.etiqueta}")
    di(f"mejor F1@3:    {mejor_f.f1:.4f}  IC [{mejor_f.ic_f1[0]:.3f}, {mejor_f.ic_f1[1]:.3f}]")
    di(f"  {nombre_de(mejor_f)}{mejor_f.etiqueta}")

    # Frente a la entrega, que es la comparación que decide algo: una receta que
    # gane a las otras recetas pero no a lo que ya está construido no se entrega.
    base = next((c for c in corridas if c.receta == "entrega"), None)
    p_de: dict[int, dict[str, float]] = {}
    if base is not None and len(corridas) > 1:
        di("")
        # Sin letras griegas: la consola de Windows en español es cp1252 y una
        # delta la mata con UnicodeEncodeError después de haber medido todo.
        di(f"{'receta':<13} {'dif.NDCG@10':>12} {'dif.F1@3':>10}   "
           f"{'veredicto':<9} {'p(NDCG)':>8} {'p(F1)':>7}")
        for i in orden:
            c = corridas[i]
            if c is base:
                continue
            dn, df = c.ndcg - base.ndcg, c.f1 - base.f1
            veredicto = "gana" if min(dn, df) > 0 else "pierde" if max(dn, df) < 0 else "mixto"
            p_de[i] = comparar_con_la_base(c, base)
            pn = p_de[i].get("p_ndcg@10")
            pf = p_de[i].get("p_f1@3")
            cols = (f"{pn:>8.4f} {pf:>7.4f}" if pn is not None else f"{'-':>8} {'-':>7}")
            di(f"{(c.receta or ''):<13} {dn:>+12.4f} {df:>+10.4f}   "
               f"{veredicto:<9} {cols}")
        # El p-valor es pareado —mismas consultas en las dos corridas— y por eso
        # distingue una ventaja pequeña y constante de una grande que viene de
        # dos consultas. Es la comparación que decide, no la diferencia de medias.
        di("  p pareado por permutación frente a la entrega; por debajo de 0.05")
        di("  la diferencia no se explica por el reparto de consultas.")
        if len(corridas) > 3:
            umbral = 0.05 / (2 * (len(corridas) - 1))
            di(f"  Son {2 * (len(corridas) - 1)} comparaciones: para leerlas todas a la vez")
            di(f"  el umbral corregido es {umbral:.4f}, no 0.05.")

    if mejor_n is not mejor_f:
        di("")
        di("  Las dos métricas premian configuraciones distintas. El reto las pondera")
        di("  por igual (Borda), pero nada obliga a usar la misma política para las dos")
        di("  listas: se puede tomar el top-10 de fragmentos de la primera y el top-3 de")
        di("  documentos de la segunda.")
    if len(corridas) > 50:
        di("")
        di(f"  Ojo: se eligió el máximo de {len(corridas)} corridas sobre las mismas 50")
        di("  consultas. Eso sobreajusta. Fíate de las diferencias que superan los")
        di("  intervalos, no de la cuarta cifra decimal.")

    carpeta.mkdir(parents=True, exist_ok=True)
    with (carpeta / "metricas.jsonl").open("w", encoding="utf-8") as fh:
        for i in orden:
            fh.write(json.dumps(
                corridas[i].to_dict(puntos[i], p_de.get(i)), ensure_ascii=False) + "\n")
    (carpeta / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    di("")
    di(f"{len(corridas)} corridas -> {carpeta}")


def listar(raiz: Path) -> int:
    """Lo mejor de cada experimento ya corrido, para compararlos."""
    carpetas = sorted(
        p for p in raiz.glob("*/metricas.jsonl")
    )
    if not carpetas:
        print(f"no hay experimentos en {raiz}")
        return 1

    print(f"{'experimento':<24} {'corridas':>8} {'NDCG@10':>8} {'F1@3':>8}  mejor por Borda")
    for ruta in carpetas:
        filas = [json.loads(l) for l in ruta.open(encoding="utf-8") if l.strip()]
        if not filas:
            continue
        mejor = filas[0]  # el archivo ya viene ordenado por Borda
        enc = "+".join(e.replace("multilingual-", "") for e in mejor["encoders"])
        # Si la corrida ganadora venía de una receta, su nombre dice más que sus
        # parámetros: es la apuesta que ganó, y está documentada.
        etiqueta = (
            f"{mejor['receta']}: " if mejor.get("receta") else ""
        ) + (
            f"{enc} {mejor['chunk']}/{mejor['solape']:.2f} {mejor['fusion']} "
            f"a={mejor['agregacion']}"
        )
        print(
            f"{ruta.parent.name:<24} {len(filas):>8} "
            f"{max(f['ndcg@10'] for f in filas):>8.4f} "
            f"{max(f['f1@3'] for f in filas):>8.4f}  {etiqueta}"
        )
    print("\nLas columnas NDCG@10 y F1@3 son el máximo de cada experimento, que")
    print("puede venir de configuraciones distintas: mira su resumen.txt.")
    return 0


# --- Entrada --------------------------------------------------------------


def lista(texto: str | None, tipo: str = "texto") -> list | None:
    """Parte una opción escrita con comas. Devuelve None si no se pidió nada.

    None y lista vacía significan cosas distintas aquí: None es «no lo pidieron,
    deja el valor que estaba» y una lista es «prueba exactamente esto».
    """
    if not texto:
        return None
    piezas = [p.strip() for p in texto.split(",") if p.strip()]
    if tipo == "entero":
        return [int(p) for p in piezas]
    if tipo == "real":
        return [float(p) for p in piezas]
    if tipo == "booleano":
        return [p.lower() in ("1", "true", "si", "sí") for p in piezas]
    if tipo == "umbral":
        # 'no' desactiva el post-filtro; un número lo fija. Las dos cosas tienen
        # que poder convivir en la misma corrida para poder compararlas.
        return [None if p.lower() in ("no", "none", "nada") else float(p) for p in piezas]
    return piezas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", help="nombre del experimento; por defecto la fecha")
    ap.add_argument("--recetas",
                    help="configuraciones completas por nombre, o 'todas'; una "
                         "corrida por receta. Ver: python -m aphelion.evaluacion.recetas")
    ap.add_argument("--preset", choices=sorted(PRESETS),
                    help="selección ya hecha para una pregunta frecuente")
    ap.add_argument("--listar", action="store_true",
                    help="muestra lo mejor de cada experimento ya corrido y sale")
    ap.add_argument("--listar-recetas", action="store_true",
                    help="muestra el catálogo de recetas con sus parámetros y sale")

    ap.add_argument("--encoders", help="lista separada por comas")
    ap.add_argument("--chunks", help="tokens por fragmento, separados por comas")
    ap.add_argument("--solapes", help="fracciones de solape, separadas por comas")
    ap.add_argument("--fusiones", help="rrf, combsum, combmnz, convexa")
    ap.add_argument("--k0", help="valores de k0 para RRF")
    ap.add_argument("--boosts", help="multiplicadores de realce por fenómeno")
    ap.add_argument("--max-por-doc", help="topes de fragmentos por documento")
    ap.add_argument("--candidatos", help="profundidad del pool por índice")
    ap.add_argument("--umbrales", help="umbrales relativos; 'no' para desactivar")
    ap.add_argument("--agregaciones", help="max, suma, media, top2, top3")
    ap.add_argument("--subdividir", help="true, false")
    ap.add_argument("--max-fusion", type=int, default=2,
                    help="cuántos encoders se pueden fusionar a la vez")

    ap.add_argument("--submuestra", type=Path, default=config.SUBMUESTRA)
    ap.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH)
    ap.add_argument("--textos", type=Path, default=config.GROUND_TRUTH_TEXTOS)
    ap.add_argument("--pruebas", type=Path, default=config.PRUEBAS)
    ap.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    ap.add_argument("--lote", type=int)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--solo-evaluar", action="store_true",
                    help="no indexa nada; usa los índices que ya existan")
    args = ap.parse_args()

    if args.listar:
        return listar(args.pruebas)

    if args.listar_recetas:
        print(mod_recetas.tabla())
        return 0

    if args.recetas and args.preset:
        print("--recetas y --preset piden cosas distintas: la receta fija todos los")
        print("parámetros y el preset abre una dimensión. Usa uno de los dos.")
        return 1

    for ruta, que in ((args.ground_truth, "ground truth"),
                      (args.textos, "textos juzgados"),
                      (args.submuestra, "submuestra")):
        if not ruta.exists():
            print(f"falta {ruta} ({que}).")
            print("  ground truth:  scripts/analisis/pool_juicios.py --consolidar")
            print("  submuestra:    scripts/analisis/submuestra.py")
            return 1

    # Dos caminos hasta la misma lista de planes: recetas completas, o la rejilla
    # de lo que se haya pedido dimensión por dimensión.
    if args.recetas:
        try:
            elegidas = mod_recetas.seleccionar(args.recetas)
        except ValueError as error:
            print(error)
            return 1
        planes = planificar_recetas(elegidas)
        # Con el motivo incluido: en seis meses, el config.json de un experimento
        # tiene que explicar por qué se probó eso y no solo qué se probó.
        detalle = {n: asdict(mod_recetas.RECETAS[n]) for n in elegidas}
        prefijo = (elegidas[0] if len(elegidas) == 1 else "recetas") + "-"
    else:
        elegidas = []
        # Las opciones: defectos, luego el preset, luego lo que se pidió por CLI.
        op = dict(DEFECTOS)
        if args.preset:
            op.update(PRESETS[args.preset])
        for clave, valor in (
            ("encoders", lista(args.encoders)),
            ("chunks", lista(args.chunks, "entero")),
            ("solapes", lista(args.solapes, "real")),
            ("fusiones", lista(args.fusiones)),
            ("k0", lista(args.k0, "entero")),
            ("boosts", lista(args.boosts, "real")),
            ("max_por_doc", lista(args.max_por_doc, "entero")),
            ("candidatos", lista(args.candidatos, "entero")),
            ("umbrales", lista(args.umbrales, "umbral")),
            ("agregaciones", lista(args.agregaciones)),
            ("subdividir", lista(args.subdividir, "booleano")),
        ):
            if valor is not None:
                op[clave] = valor
        op["max_fusion"] = args.max_fusion

        desconocidos = [e for e in op["encoders"] if e not in config.ENCODERS]
        if desconocidos:
            print(f"encoders desconocidos: {desconocidos}")
            print(f"disponibles: {sorted(config.ENCODERS)}")
            return 1

        planes = planificar_rejilla(op)
        detalle = {k: v for k, v in op.items()}
        prefijo = (args.preset + "-") if args.preset else ""

    nombre = args.nombre or (prefijo + datetime.now().strftime("%Y%m%d-%H%M"))
    carpeta = args.pruebas / nombre

    # Agrupadas por fragmentación: los índices de un chunking se cargan una vez y
    # todas sus corridas se miden sobre los mismos rankings en memoria.
    por_chunking: dict[tuple[int, float], list[Plan]] = {}
    for plan in planes:
        por_chunking.setdefault((plan.chunk, plan.solape), []).append(plan)

    pares = pares_a_indexar(planes)
    faltan_indices = sum(
        1
        for (chunk, solape), encs in pares.items()
        for e in encs
        if not (ruta_indices(chunk, solape) / f"encoder_{e}" / "index.faiss").exists()
    )

    print(f"experimento: {nombre}  ->  {carpeta}")
    if elegidas:
        for nombre_receta in elegidas:
            receta = mod_recetas.RECETAS[nombre_receta]
            print(f"  {nombre_receta:<13} {receta.resumen()}")
            print(f"  {'':<13} {receta.apuesta}")
            for fallo in receta.problemas():
                print(f"  {'':<13} ! {fallo}")
    else:
        print(f"  encoders:     {op['encoders']}  (fusionando hasta {op['max_fusion']})")
        print(f"  chunkings:    {[clave_chunking(*c) for c in por_chunking]}")
        print(f"  fusiones:     {op['fusiones']}   k0={op['k0']}")
        print(f"  agregaciones: {op['agregaciones']}")
        print(f"  realce:       {op['boosts']}   por doc={op['max_por_doc']}   "
              f"pool={op['candidatos']}   umbral={op['umbrales']}")
    print(f"  a indexar:    {faltan_indices} índices en "
          f"{len(pares)} fragmentaci{'ón' if len(pares) == 1 else 'ones'}")
    print(f"  a evaluar:    {len(planes):,} corridas  (~{len(planes) * 0.2:.0f} min)")
    avisar_de_ventanas(planes)
    print()

    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "config.json").write_text(
        json.dumps(
            {
                "nombre": nombre, "preset": args.preset, "recetas": elegidas,
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "corridas": len(planes),
                "opciones": detalle,
                "submuestra": str(args.submuestra),
                "ground_truth": str(args.ground_truth),
                "umbral_emparejamiento": emparejamiento.UMBRAL,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )

    if not args.solo_evaluar:
        t0 = time.time()
        preparar(planes, args.submuestra, args.backend, args.lote)
        print(f"\nindexación lista en {(time.time() - t0) / 60:.1f} min\n")

    juicios = metricas.cargar_juicios(args.ground_truth)
    textos = emparejamiento.cargar_textos(args.textos)
    empar = emparejamiento.emparejadores(juicios, textos)
    preguntas = [p for p in mod_consultas.cargar() if p.query_id in juicios]
    print(f"ground truth: {len(juicios)} consultas, {len(textos):,} textos juzgados")
    print(f"emparejamiento de fragmentos por texto, umbral {emparejamiento.UMBRAL}\n")

    corridas: list[Corrida] = []
    for (chunk, solape), grupo in por_chunking.items():
        print(f"{clave_chunking(chunk, solape)}:")
        corridas.extend(
            evaluar_chunking(chunk, solape, grupo, juicios, empar, preguntas)
        )

    informar(corridas, args.top, carpeta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
