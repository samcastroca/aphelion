"""Exporta un informe en Markdown a PDF sin depender de pandoc.

`scripts/etapas/05_empaquetar.py` usa pandoc, que es la herramienta correcta
cuando está. No siempre está: es una instalación aparte del entorno de Python y
`uv sync` no la trae, así que en una máquina nueva el informe se queda sin
exportar y la entrega sale incompleta por una dependencia externa.

Esto lo resuelve con lo que ya está en el entorno: `markdown-it-py` para pasar a
HTML y el motor de composición `Story` de PyMuPDF para paginar. No pretende
sustituir a pandoc en general —no hace bibliografía, ni notas al pie, ni
referencias cruzadas—; hace lo que este informe necesita: encabezados, párrafos,
listas, tablas, bloques de código y citas.

El reto limita el documento técnico a 8 páginas (§1.4), así que el script cuenta
las páginas que produjo y avisa si se pasa, en vez de dejar que el exceso se
descubra al entregar.

Uso:
    uv run python scripts/informe_a_pdf.py docs/informe_tecnico_v2.md \\
        --salida entrega-v2/informe_tecnico.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_PAGINAS = 8

# Hoja de estilo del documento. Va aquí y no en el Markdown porque el Markdown es
# la fuente que se lee y se versiona; el aspecto es cosa de la exportación.
CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.35; }
h1 { font-size: 16pt; margin: 0 0 2pt 0; }
h2 { font-size: 12pt; margin: 12pt 0 3pt 0; }
h3 { font-size: 10.5pt; margin: 9pt 0 2pt 0; }
p  { margin: 0 0 5pt 0; text-align: justify; }
li { margin: 0 0 2pt 0; }
code { font-family: monospace; font-size: 8.5pt; }
pre  { font-family: monospace; font-size: 7.5pt; line-height: 1.15;
       background: #f4f4f4; padding: 4pt; margin: 0 0 6pt 0; }
table { font-size: 8.5pt; margin: 0 0 6pt 0; }
th { text-align: left; background: #eeeeee; padding: 2pt 5pt; }
td { padding: 2pt 5pt; }
blockquote { margin: 0 0 6pt 10pt; font-style: italic; }
"""


def sin_portada_yaml(texto: str) -> str:
    """Quita el bloque YAML inicial, que es para pandoc y no para el cuerpo."""
    if texto.startswith("---"):
        fin = texto.find("\n---", 3)
        if fin != -1:
            return texto[fin + 4 :].lstrip("\n")
    return texto


def a_html(markdown: str) -> str:
    from markdown_it import MarkdownIt

    # `tables` no viene en el preajuste de CommonMark y el informe las usa para
    # las mediciones, que son medio documento.
    md = MarkdownIt("commonmark").enable("table")
    cuerpo = md.render(markdown)

    # Story compone cada bloque por separado y no hereda `font-family` a las
    # celdas; se deja el CSS al motor y aquí solo va la estructura.
    return f"<html><head><style>{CSS}</style></head><body>{cuerpo}</body></html>"


def exportar(origen: Path, destino: Path, ancho: float, alto: float) -> int:
    import pymupdf

    markdown = sin_portada_yaml(origen.read_text(encoding="utf-8"))
    html = a_html(markdown)

    destino.parent.mkdir(parents=True, exist_ok=True)
    historia = pymupdf.Story(html=html, user_css=None)
    marco = pymupdf.Rect(0, 0, ancho, alto)
    # Márgenes de 2 cm, en puntos.
    caja = marco + (56, 56, -56, -56)

    escritor = pymupdf.DocumentWriter(destino)
    paginas = 0
    mas = True
    while mas:
        dispositivo = escritor.begin_page(marco)
        mas, _ = historia.place(caja)
        historia.draw(dispositivo)
        escritor.end_page()
        paginas += 1
        if paginas > 50:  # cinturón: un HTML mal formado no debe iterar sin fin
            raise RuntimeError("la composición no termina; revisa el Markdown")
    escritor.close()
    return paginas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("origen", type=Path)
    ap.add_argument("--salida", type=Path, required=True)
    # A4 en puntos.
    ap.add_argument("--ancho", type=float, default=595.0)
    ap.add_argument("--alto", type=float, default=842.0)
    args = ap.parse_args()

    if not args.origen.exists():
        print(f"no existe {args.origen}")
        return 1

    paginas = exportar(args.origen, args.salida, args.ancho, args.alto)
    print(f"-> {args.salida}  ({paginas} páginas)")

    if paginas > MAX_PAGINAS:
        print(f"\nERROR: el reto limita el documento técnico a {MAX_PAGINAS} "
              f"páginas (§1.4) y salieron {paginas}.")
        print("Recorta el Markdown; no se entrega un informe que incumple el límite.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
