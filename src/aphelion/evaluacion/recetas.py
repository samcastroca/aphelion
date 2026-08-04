"""Configuraciones completas con las que vale la pena competir.

Una **receta** fija *todas* las dimensiones a la vez: qué encoders, con qué
fragmentación, cómo fusionar, cómo agregar a documentos y qué políticas de
realce, diversificación, profundidad y umbral. Eso la distingue de un preset,
que abre una dimensión y deja las demás quietas para aislar un efecto. El preset
sirve para *entender*; la receta sirve para *competir*: es una candidata a ser la
configuración que se entrega, y se mide contra las otras candidatas en la misma
tabla y con el mismo Conteo de Borda.

Cada receta es **una corrida**. Pedir cinco recetas son cinco corridas, no un
producto cartesiano, así que la tabla que sale se lee de arriba abajo sin tener
que descontar el sobreajuste de haber probado miles de variantes.

**De dónde sale cada apuesta.** Ninguna es un tanteo: o replica lo que se
entrega, o parte de algo medido en este corpus (`docs/DISENO.md` §9.bis) que
sugiere que se puede mejorar. Lo que ninguna receta hace es cambiar dos cosas a
la vez sin motivo, porque entonces el resultado no dice cuál de las dos ganó.

    uv run python -m aphelion.evaluacion.recetas          # ver el catálogo
    uv run python scripts/analisis/barrido_completo.py --recetas entrega,dos-encoders
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class Receta:
    """Una configuración completa, sin nada que quede al valor por defecto."""

    apuesta: str  # una línea: qué se juega esta receta
    por_que: str  # en qué se basa; sale al informe y al config.json
    encoders: tuple[str, ...]
    chunk: int
    solape: float
    fusion: str
    k0: int
    boost: float
    max_por_doc: int
    candidatos: int
    umbral: float | None
    subdividir: bool
    agregacion: str

    def politica(self) -> dict:
        """Las claves que consume el barrido para una corrida."""
        return {
            "fusion": self.fusion,
            "k0": self.k0,
            "boost": self.boost,
            "max_por_doc": self.max_por_doc,
            "candidatos": self.candidatos,
            "umbral": self.umbral,
            "subdividir": self.subdividir,
            "agregacion": self.agregacion,
        }

    @property
    def sola(self) -> bool:
        return len(self.encoders) == 1

    def resumen(self) -> str:
        """Una línea con lo que define la receta, para el menú y la tabla.

        Se enseñan siempre los encoders, la fragmentación, la fusión y la
        agregación —son lo que decide una receta— y solo los demás parámetros
        que se aparten de la entrega, para que la línea quepa en una terminal y
        lo que se aparta salte a la vista.
        """
        enc = "+".join(e.replace("multilingual-", "") for e in self.encoders)
        # Con un solo índice no hay nada que fusionar y nombrar una fusión ahí
        # haría creer que se está midiendo algo que no interviene.
        fusion = "sin fusión" if self.sola else self.fusion
        linea = f"{enc:<22} {self.chunk}/{self.solape:.2f}  {fusion:<11} a={self.agregacion:<5}"

        extras = []
        if not self.sola and self.fusion == "rrf" and self.k0 != config.RRF_K0:
            extras.append(f"k0={self.k0}")
        if self.boost != config.BOOST_FENOMENO:
            extras.append(f"realce={self.boost:g}")
        if self.max_por_doc != config.MAX_FRAGMENTOS_POR_DOC:
            extras.append(f"d={self.max_por_doc}")
        if self.candidatos != config.CANDIDATOS_POR_INDICE:
            extras.append(f"pool={self.candidatos}")
        if self.umbral is not None:
            extras.append(f"umbral={self.umbral:g}")
        if self.subdividir:
            extras.append("subdivide")
        return (linea + "  " + " ".join(extras)).rstrip()

    def indices(self) -> list[tuple[int, float, str]]:
        """Los (chunk, solape, encoder) que hay que tener indexados."""
        return [(self.chunk, self.solape, e) for e in self.encoders]

    def faltantes(self) -> int:
        """Cuántos de sus índices no están todavía en pruebas\\indices."""
        base = config.ruta_indices_prueba(self.chunk, self.solape)
        return sum(
            0 if (base / f"encoder_{e}" / "index.faiss").exists() else 1
            for e in self.encoders
        )

    def problemas(self) -> list[str]:
        """Lo que haría que esta receta midiera algo distinto de lo que dice."""
        fallos = []
        for encoder in self.encoders:
            if encoder not in config.ENCODERS:
                fallos.append(f"{encoder} no está en el catálogo")
            elif not config.cabe_en_ventana(self.chunk, encoder):
                ventana = config.ENCODERS[encoder]["max_tokens"]
                fallos.append(
                    f"{self.chunk} tokens no caben en {encoder} (ventana {ventana}): "
                    "se truncaría en silencio"
                )
        if self.sola and self.fusion != "rrf":
            fallos.append("un solo encoder: la fusión no interviene, ponla en rrf")
        return fallos


# La entrega, para que todas las demás se midan contra algo y no entre ellas.
_ENTREGA = dict(
    chunk=config.CHUNK_PRESUPUESTO,
    solape=config.CHUNK_SOLAPE,
    fusion="rrf",
    k0=config.RRF_K0,
    boost=config.BOOST_FENOMENO,
    max_por_doc=config.MAX_FRAGMENTOS_POR_DOC,
    candidatos=config.CANDIDATOS_POR_INDICE,
    umbral=None,
    subdividir=False,
    agregacion=config.AGREGACION_DOCUMENTOS,
)


def _variante(**cambios) -> Receta:
    """Una receta descrita por lo que cambia respecto a la entrega.

    Escribirlas así evita el error de copiar doce parámetros y equivocarse en
    uno: lo que no se nombra es, por construcción, idéntico a lo que se entrega,
    y la comparación mide solo lo que se nombró.
    """
    return Receta(**{**_ENTREGA, **cambios})


RECETAS: dict[str, Receta] = {
    "entrega": _variante(
        apuesta="la línea base: exactamente lo que se entrega hoy",
        por_que=(
            "No es una candidata, es la vara de medir. Sin ella una tabla de "
            "recetas dice cuál es la mejor entre las nuevas, que es la pregunta "
            "equivocada: lo que hay que saber es si alguna le gana a lo que ya "
            "está construido. "
            "Desde que se midió sobre el corpus completo, la entrega es BGE-M3 "
            "solo con agregación top2: le ganaba a los dos encoders con max por "
            "+0,1074 de NDCG@10 (p=0,0002, pareado sobre el entregable real) sin "
            "ceder nada medible en F1@3."
        ),
        encoders=tuple(config.ENCODERS_ENTREGA),
    ),
    "dos-encoders": _variante(
        apuesta="lo que se entregaba antes: fusionar BGE-M3 con mE5-large por RRF",
        por_que=(
            "Era la entrega hasta que el corpus completo dijo otra cosa. Se queda "
            "en el catálogo porque retirar un encoder es la decisión más cara de "
            "revertir —hay que recodificar 64.484 fragmentos— y conviene poder "
            "volver a medirla cuando cambie el ground truth o la fragmentación, "
            "sin reconstruirla de memoria."
        ),
        encoders=("bge-m3", "me5-large"),
        agregacion="max",
    ),
    "convexa": _variante(
        apuesta="fusionar por similitud normalizada en vez de por posición",
        por_que=(
            "RRF tira la magnitud: un candidato que gana por 0,30 de coseno y otro "
            "que gana por 0,01 aportan lo mismo. La combinación convexa de "
            "puntuaciones normalizadas por min-max conserva esa distancia (Bruch "
            "et al. 2022) y es la alternativa que el diseño dejó abierta sin "
            "medir. Con top2 para no cambiar dos cosas a la vez respecto a "
            "`entrega`, que es la receta con la que compite."
        ),
        encoders=("bge-m3", "me5-large"),
        fusion="convexa",
        agregacion="top2",
    ),
    "familias": _variante(
        apuesta="fusionar dos familias distintas en vez de dos parientes",
        por_que=(
            "BGE-M3 y mE5 salen los dos de XLM-RoBERTa, así que se equivocan en "
            "cosas parecidas y fusionarlos aporta menos consenso del que aparenta. "
            "GTE-multilingual-base viene de otro preentrenamiento: si la fusión "
            "vale por decorrelación de errores, este par debería ganarle al par de "
            "la entrega. Si no le gana, la fusión estaba aportando poco y "
            "la entrega de un solo encoder se refuerza."
        ),
        encoders=("bge-m3", "gte-multilingual-base"),
        agregacion="top2",
    ),
    "filtrado": _variante(
        apuesta="descartar la cola floja de cada índice antes de fusionar",
        por_que=(
            "El post-filtro de la §8.7 quita de cada índice los candidatos por "
            "debajo del 90% de la mejor similitud de esa consulta. Está "
            "implementado y desactivado porque el valor tenía que salir del "
            "barrido y no de una corazonada. Debería ayudar al F1@3, donde un "
            "documento sostenido por un solo fragmento mediocre gasta una de las "
            "tres posiciones; puede hacer daño al NDCG@10 si deja el top-10 "
            "incompleto, y por eso hay que medirlo y no suponerlo."
        ),
        encoders=("bge-m3", "me5-large"),
        umbral=0.9,
        agregacion="top2",
    ),
    "barato": _variante(
        apuesta="¿hace falta un modelo grande?",
        por_que=(
            "mE5-base son 768 dimensiones contra 1024 y codifica varias veces más "
            "rápido: en la GPU de 4 GB donde BGE-M3 va a 2,6 fragmentos por "
            "segundo, eso es la diferencia entre una tarde y una noche. Si la "
            "brecha de métricas es menor que los intervalos de confianza, el "
            "modelo grande no se está pagando. Es la receta que puede cambiar cómo "
            "se trabaja, no solo qué se entrega."
        ),
        encoders=("me5-base",),
        agregacion="top2",
    ),
    # --- De aquí abajo cambian la fragmentación: hay que re-fragmentar y
    # re-codificar la submuestra entera, y no se reutiliza el índice de las de
    # arriba. Van al final para que una corrida interrumpida deje hecho lo que
    # comparte con las demás.
    "sin-recorte": _variante(
        apuesta="fragmentos que quepan enteros en las 250 palabras del reto",
        por_que=(
            "El presupuesto de 504 tokens no mira el límite de palabras: medido "
            "sobre la submuestra, el 79,9% de los fragmentos lo excede (93,2% de "
            "los que están en español) y se entregan truncados, perdiendo ~20% de "
            "la cola. La recuperación no sufre —el vector se calculó sobre el "
            "fragmento entero— pero el jurado empareja sobre el texto entregado "
            "(§10.2.1), así que la frase que justificó el acierto puede haberse "
            "quedado fuera. A 345 tokens el 90% cabe sin recortar. Lo que se "
            "arriesga a cambio es contexto por vector, y eso es lo que mide."
        ),
        encoders=("bge-m3",),
        chunk=345,
        agregacion="top2",
    ),
    "granular": _variante(
        apuesta="fragmentos cortos y sin solape, apostando al NDCG@10",
        por_que=(
            "El NDCG@10 premia precisión en diez posiciones: un fragmento de 256 "
            "tokens (~160 palabras) es más específico y no se recorta nunca. Sube "
            "el tope por documento a 5 porque al partir más fino la evidencia de un "
            "mismo informe se reparte entre más fragmentos y el tope de 3 empezaría "
            "a descartar aciertos. Sin solape porque cuesta un 15% de vectores y no "
            "hay evidencia de que aporte."
        ),
        encoders=("bge-m3",),
        chunk=256,
        solape=0.0,
        max_por_doc=5,
        agregacion="max",
    ),
    "contexto": _variante(
        apuesta="fragmentos largos y top3, apostando al F1@3",
        por_que=(
            "La apuesta contraria a `granular`, y sobre la otra métrica: para "
            "acertar el documento conviene que cada vector resuma más del "
            "documento. Solo BGE-M3, porque 768 tokens no caben en la ventana de "
            "512 de mE5 y ahí se truncarían sin avisar. top3 y tope 5 por la misma "
            "razón: con fragmentos largos, tres buenos del mismo documento son "
            "mejor señal de que el documento es el correcto que uno solo."
        ),
        encoders=("bge-m3",),
        chunk=768,
        max_por_doc=5,
        agregacion="top3",
    ),
}


# --- Selección ------------------------------------------------------------


def seleccionar(texto: str) -> list[str]:
    """'entrega,convexa' o 'todas' -> nombres de receta, en orden de catálogo.

    Respeta el orden del catálogo y no el de escritura: las que comparten la
    fragmentación de la entrega van primero y las que obligan a re-fragmentar al
    final, así que una corrida interrumpida deja hecho lo que se reutiliza.
    """
    if texto.strip().lower() in ("todas", "todo", "*"):
        return list(RECETAS)
    pedidas = {p.strip() for p in texto.split(",") if p.strip()}
    desconocidas = pedidas - set(RECETAS)
    if desconocidas:
        raise ValueError(
            f"recetas desconocidas: {sorted(desconocidas)}. "
            f"Hay: {', '.join(RECETAS)}"
        )
    return [n for n in RECETAS if n in pedidas]


def tabla() -> str:
    """El catálogo con sus parámetros, para leerlo desde la terminal."""
    lineas = [
        f"{'receta':<13} {'configuración':<66} índices",
        "",
    ]
    for nombre, receta in RECETAS.items():
        faltan = receta.faltantes()
        coste = "reutiliza" if not faltan else f"indexa {faltan}"
        lineas.append(f"{nombre:<13} {receta.resumen():<66} {coste}")
        lineas.append(f"{'':<13} {receta.apuesta}")
        for fallo in receta.problemas():
            lineas.append(f"{'':<13} ! {fallo}")
    lineas += [
        "",
        "Cada receta es una corrida. 'configuración' enseña encoders, "
        "tokens/solape,",
        "fusión y agregación; lo que no aparece es igual que en la entrega.",
    ]
    return "\n".join(lineas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tsv",
        action="store_true",
        help="una línea por receta separada por tabuladores, para el menú",
    )
    args = ap.parse_args()

    if args.tsv:
        for nombre, receta in RECETAS.items():
            faltan = receta.faltantes()
            coste = "reutiliza índices" if not faltan else f"indexa {faltan}"
            print("\t".join((nombre, receta.resumen(), receta.apuesta, coste)))
        return 0

    print(tabla())
    return 0


if __name__ == "__main__":
    sys.exit(main())
