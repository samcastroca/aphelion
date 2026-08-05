"""Detección de la estructura de un documento: dónde empieza y acaba cada sección.

**Por qué existe.** La extracción guarda texto plano: `_extraer_pdf` llama a
`get_text("text", sort=True)` y de ahí en adelante el tamaño de fuente, la
negrita y la posición ya no están en ninguna parte. Para partir por secciones hay
que volver al PDF original, que es lo que hace este módulo. Se llega a él por
`config.CORPUS / doc["ruta_rel"]`, la ruta relativa que la ingesta sí conserva.

**Qué cuenta como encabezado.** No hay marcas semánticas en un PDF: hay glifos con
un tamaño y unos atributos. Se combinan tres señales, todas relativas al propio
documento y no a umbrales absolutos, porque el cuerpo de un atlas RESDAL y el de
un informe CSET no miden lo mismo:

1. **Tamaño mayor que el del cuerpo.** El tamaño del cuerpo se estima como el más
   frecuente ponderado por caracteres, no por número de spans: un documento con
   cien títulos cortos y diez párrafos largos tiene más spans de título que de
   texto, y contar spans elegiría el título como cuerpo.
2. **Negrita a tamaño de cuerpo.** Muchos informes marcan los epígrafes así.
3. **Línea corta que no termina en punto.** Descarta el párrafo entero en negrita
   y la primera línea de un texto justificado, que empieza grande por capitular.

Un documento sin al menos dos encabezados se declara sin estructura utilizable y
quien llama vuelve a la estrategia fija. Es el caso de los 60 PDFs escaneados:
sin capa de texto no hay spans, así que `secciones_pdf` devuelve None por el
mismo camino que un PDF ilegible, y su texto reconocido por OCR se fragmenta como
siempre.

**HTML queda fuera a propósito.** El corpus no tiene ni un archivo HTML —son 759
PDF, 954 JSON, 26 CSV, 4 XLSX, 74 PBF, 8 imágenes y 1 TXT—, así que una rama
h1–h4 sería código que ninguna corrida ejercita. El punto de extensión es
`secciones_de`: añadir un formato es añadirle una rama.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Un encabezado tiene que ser mayor que el cuerpo por un margen: en muchos PDFs
# conviven 10,0 y 10,04 por redondeo del motor de composición, y sin margen cada
# párrafo se convertiría en una sección.
FACTOR_TITULO = 1.15

# Una línea más larga que esto es un párrafo aunque venga en negrita o en cuerpo
# mayor. El valor está en caracteres y no en tokens porque aquí todavía no hay
# tokenizador: esta fase decide fronteras, no presupuestos.
MAX_CARACTERES_TITULO = 140

# Menos de dos encabezados no es una estructura, es un título de portada.
MIN_ENCABEZADOS = 2

# Bit de negrita en los `flags` de un span de PyMuPDF.
_NEGRITA = 1 << 4


@dataclass(frozen=True)
class Seccion:
    """Un tramo de documento con su título, si lo tenía."""

    titulo: str | None
    texto: str
    nivel: int  # 1 es el encabezado mayor del documento; 0, el texto sin título


def _lineas_con_formato(pdf) -> list[dict]:
    """Las líneas del documento con el formato que las compone.

    Se pide `sort=True` por la misma razón que la extracción: reordena los
    bloques según el flujo de lectura, sin lo cual un documento a dos columnas
    —los atlas RESDAL, los informes CSIS— entrega las líneas intercaladas entre
    columnas y ninguna heurística de encabezados sobrevive a eso.
    """
    lineas: list[dict] = []
    for pagina in pdf:
        bloques = pagina.get_text("dict", sort=True).get("blocks", [])
        for bloque in bloques:
            for linea in bloque.get("lines", []):
                spans = [s for s in linea.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                texto = "".join(s["text"] for s in spans).strip()
                if not texto:
                    continue
                # El span que más caracteres aporta manda: una línea de cuerpo
                # con una palabra en negrita sigue siendo cuerpo.
                dominante = max(spans, key=lambda s: len(s["text"]))
                lineas.append({
                    "texto": texto,
                    "tam": round(float(dominante["size"]), 1),
                    "negrita": bool(int(dominante["flags"]) & _NEGRITA),
                })
    return lineas


def _tamano_cuerpo(lineas: list[dict]) -> float:
    """El tamaño de fuente del texto corrido, ponderado por caracteres."""
    peso: Counter = Counter()
    for linea in lineas:
        peso[linea["tam"]] += len(linea["texto"])
    return peso.most_common(1)[0][0] if peso else 0.0


# Con qué puede terminar una línea para que lo siguiente empiece de cero. Si la
# anterior no cierra aquí, lo que sigue es la continuación de una frase.
FIN_LIMPIO = (".", "!", "?", ":", "…", "”", "\"", "»", ")")


def _cierra_frase(texto: str) -> bool:
    return texto.rstrip().endswith(FIN_LIMPIO)


def _es_encabezado(linea: dict, cuerpo: float, previa: dict | None) -> bool:
    if len(linea["texto"]) > MAX_CARACTERES_TITULO:
        return False
    # Una línea que cierra en punto es prosa: los títulos no se puntúan.
    if linea["texto"].endswith((".", ";", ",")):
        return False
    # Folios, numeración suelta y viñetas huérfanas no son secciones.
    if not any(c.isalpha() for c in linea["texto"]):
        return False

    # Un salto de tamaño es señal por sí solo: nadie compone media frase en
    # cuerpo 20 dentro de un párrafo en cuerpo 10.
    if linea["tam"] > cuerpo * FACTOR_TITULO:
        return True

    # La negrita a tamaño de cuerpo, en cambio, aparece dentro del texto corrido
    # —términos destacados, entradas de glosario— y ahí no abre sección. Solo
    # cuenta si lo anterior cerró: si la línea previa quedó a medias, esto es su
    # continuación, no un epígrafe. Sin esta condición, un informe con negritas
    # intercaladas se parte a mitad de frase una y otra vez.
    if not (linea["negrita"] and linea["tam"] >= cuerpo):
        return False
    return previa is None or _cierra_frase(previa["texto"])


def secciones_pdf(ruta: Path) -> list[Seccion] | None:
    """Las secciones de un PDF, o None si no tiene estructura utilizable.

    Devolver None y no una lista de una sección es deliberado: quien llama tiene
    que poder distinguir «este documento es una sección larga» de «no sé leer
    este documento», porque en el segundo caso la respuesta correcta es la
    estrategia fija y no un fragmento gigante.
    """
    try:
        import pymupdf

        with pymupdf.open(ruta) as pdf:
            lineas = _lineas_con_formato(pdf)
    except Exception:
        # Un PDF ilegible no puede tumbar la fragmentación del corpus: se cae a
        # la estrategia fija, que trabaja sobre el texto ya extraído.
        return None

    if not lineas:
        return None  # escaneado: sin capa de texto no hay tipografía

    cuerpo = _tamano_cuerpo(lineas)
    if cuerpo <= 0:
        return None

    marcas = [
        i
        for i, l in enumerate(lineas)
        if _es_encabezado(l, cuerpo, lineas[i - 1] if i else None)
    ]
    if len(marcas) < MIN_ENCABEZADOS:
        return None

    # Nivel por rango de tamaño entre los encabezados detectados: el mayor es 1.
    tamanos = sorted({lineas[i]["tam"] for i in marcas}, reverse=True)
    nivel_de = {t: i + 1 for i, t in enumerate(tamanos)}

    secciones: list[Seccion] = []

    def cerrar(titulo: str | None, cuerpo_lineas: list[str], nivel: int) -> None:
        texto = "\n".join(cuerpo_lineas).strip()
        if texto:
            secciones.append(Seccion(titulo=titulo, texto=texto, nivel=nivel))

    # Lo que va antes del primer encabezado es sección sin título: portada,
    # resumen, datos de publicación. Descartarlo perdería texto recuperable.
    inicio = marcas[0]
    cerrar(None, [l["texto"] for l in lineas[:inicio]], 0)

    for orden, i in enumerate(marcas):
        fin = marcas[orden + 1] if orden + 1 < len(marcas) else len(lineas)
        titulo = lineas[i]["texto"]
        # El título entra también en el texto de la sección: es lo que la nombra
        # y su vocabulario es justo el que la consulta suele traer.
        cerrar(
            titulo,
            [titulo] + [l["texto"] for l in lineas[i + 1 : fin]],
            nivel_de[lineas[i]["tam"]],
        )

    return secciones or None


def secciones_de(doc: dict, formato: str, raiz_corpus: Path) -> list[Seccion] | None:
    """Punto único de entrada: las secciones de un documento, o None.

    Solo PDF por ahora. JSON, CSV, XLSX, PBF y TXT devuelven None y su
    fragmentación cae a la estrategia fija, que es lo correcto: sus formatos no
    declaran jerarquía, y la que se les inventara sería ruido con forma de
    estructura.
    """
    if formato != "pdf":
        return None

    ruta_rel = doc.get("ruta_rel") or ""
    if not ruta_rel:
        return None

    ruta = raiz_corpus / ruta_rel
    if not ruta.exists():
        # Pasa cuando se fragmenta en una máquina sin el corpus de 3 GB. No es un
        # error: significa que este documento va por la estrategia fija.
        return None

    return secciones_pdf(ruta)
