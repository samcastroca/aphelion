"""Aplica OCR a los documentos que `01_extraer.py` dejó sin capa de texto.

Cuáles son se lee de los propios JSON de `trabajo/texto/`, donde la extracción
los marcó con `necesita_ocr`. Reescribe el JSON de cada uno con el texto
reconocido, de modo que `03_fragmentar.py` no necesita saber que el texto vino de
un escaneo. El resto del pipeline es idéntico.

Sobre cada salida se calculan tres señales de calidad, porque un OCR que falla en
silencio es peor que uno que revienta:

- Densidad de diacríticos, **solo cuando el texto resulta estar en español**. Un
  texto español sin tildes ni eñes indica que Tesseract leyó con el paquete de
  idioma equivocado, aunque el resultado parezca legible. Los pendientes no son
  todos españoles —48 son de la Defensoría, el resto de CSET y CSIS—, así que el
  idioma se detecta sobre la salida en lugar de asumirse.
- Proporción de caracteres no imprimibles, que delata ruido de rasterización.
- Palabras por página, que delata páginas en blanco o rasterizadas mal.

Uso:
    uv run python scripts/etapas/02_ocr.py --muestra 2 --paginas 3   # inspección previa
    uv run python scripts/etapas/02_ocr.py                           # el lote completo
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata

from tqdm import tqdm

from aphelion import config
from aphelion.ingesta import limpieza, ocr

_DIACRITICOS = set("áéíóúñüÁÉÍÓÚÑÜ")

# Por debajo de esto el texto no es español reconocido correctamente. Medido
# sobre prosa institucional en español, la densidad típica está en 0.02–0.06.
UMBRAL_DIACRITICOS = 0.005
UMBRAL_NO_IMPRIMIBLE = 0.01
MIN_PALABRAS_POR_PAGINA = 20


def densidad_diacriticos(texto: str) -> float:
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return 0.0
    return sum(1 for c in letras if c in _DIACRITICOS) / len(letras)


def ratio_no_imprimible(texto: str) -> float:
    if not texto:
        return 0.0
    malos = sum(
        1
        for c in texto
        if unicodedata.category(c) in ("Cc", "Co", "Cn") and c not in "\n\t"
    )
    return malos / len(texto)


def senales(resultado: ocr.ResultadoOCR) -> dict:
    texto = resultado.texto
    palabras = len(texto.split())
    paginas = max(resultado.num_paginas, 1)
    return {
        "paginas": resultado.num_paginas,
        "palabras": palabras,
        "palabras_por_pagina": round(palabras / paginas, 1),
        "idioma": limpieza.detectar_idioma(texto),
        "diacriticos": round(densidad_diacriticos(texto), 4),
        "no_imprimible": round(ratio_no_imprimible(texto), 5),
        "segundos": round(resultado.segundos, 1),
    }


def sospechoso(s: dict) -> list[str]:
    avisos = []
    # Solo tiene sentido exigir tildes si el texto salió en español: entre los
    # pendientes hay informes de CSET y CSIS, que están en inglés.
    if s["idioma"] == "es" and s["diacriticos"] < UMBRAL_DIACRITICOS:
        avisos.append(f"español sin diacríticos ({s['diacriticos']:.4f})")
    if s["no_imprimible"] > UMBRAL_NO_IMPRIMIBLE:
        avisos.append(f"ruido ({s['no_imprimible']:.4f} no imprimible)")
    if s["palabras_por_pagina"] < MIN_PALABRAS_POR_PAGINA:
        avisos.append(f"casi vacío ({s['palabras_por_pagina']} palabras/página)")
    return avisos


def pendientes() -> list[str]:
    """Los documentos sin capa de texto, leídos de la caché de extracción.

    Cuáles son es un hecho sobre `trabajo/texto/`, no sobre la corrida que lo
    llenó. Cuando esto dependía de una lista que `01_extraer.py` escribía con lo
    que había procesado, bastaba con reanudar el pipeline —extracción cacheada,
    nada que procesar, lista vacía— para que esta etapa se quedara sin entrada.

    Cuesta releer los JSON del corpus entero, unos segundos, y es barato al lado
    de las horas de Tesseract que vienen después.
    """
    ids: list[str] = []
    for ruta in sorted(config.TEXTO_CRUDO.glob("*.json")):
        registro = json.loads(ruta.read_text(encoding="utf-8"))
        if registro.get("necesita_ocr"):
            ids.append(registro.get("doc_id") or ruta.stem)
    return ids


def leer_cache() -> dict[str, dict]:
    if not config.OCR_CACHE.exists():
        return {}
    with config.OCR_CACHE.open(encoding="utf-8") as fh:
        return {r["doc_id"]: r for r in (json.loads(l) for l in fh if l.strip())}


def escribir_cache(cache: dict[str, dict]) -> None:
    config.OCR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with config.OCR_CACHE.open("w", encoding="utf-8") as fh:
        for doc_id in sorted(cache):
            fh.write(json.dumps(cache[doc_id], ensure_ascii=False) + "\n")


def aplicar(destino, registro: dict, texto: str, senales_ocr: dict) -> None:
    registro["texto"] = texto
    registro["necesita_ocr"] = False
    registro["meta"] = {**registro.get("meta", {}), "ocr": senales_ocr}
    destino.write_text(json.dumps(registro, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paginas", type=int, help="máximo de páginas por PDF")
    ap.add_argument("--muestra", type=int, help="procesa solo los N primeros")
    ap.add_argument("--dpi", type=int, default=ocr.DPI)
    ap.add_argument("--forzar", action="store_true", help="reprocesa lo ya hecho")
    ap.add_argument(
        "--sin-cache", action="store_true", help="ignora data/ocr.jsonl y reconoce todo"
    )
    args = ap.parse_args()

    if not config.TEXTO_CRUDO.exists():
        print(f"No existe {config.TEXTO_CRUDO}. Ejecuta antes scripts/etapas/01_extraer.py")
        return 1

    doc_ids = pendientes()
    if not doc_ids:
        print("ningún documento pendiente de OCR")
        return 0
    if args.muestra:
        doc_ids = doc_ids[: args.muestra]

    cache = {} if args.sin_cache else leer_cache()
    por_reconocer = [d for d in doc_ids if args.forzar or d not in cache]
    print(f"pendientes: {len(doc_ids)}  |  en caché: {len(doc_ids) - len(por_reconocer)}")

    # Tesseract solo hace falta si queda algo por reconocer. Así la máquina que
    # únicamente indexa no necesita instalarlo.
    #
    # Que falte no puede impedir aplicar la caché: son los 48 escaneados de la
    # Defensoría que responden q033–q050, viajan versionados justamente para no
    # depender de Tesseract, y abortar aquí los dejaría fuera del índice — que es
    # el daño que esta etapa existe para evitar.
    hay_tesseract, detalle = (True, "") if not por_reconocer else ocr.disponible()
    if por_reconocer and hay_tesseract:
        print(f"tesseract {detalle}, idioma {ocr.IDIOMA}, {args.dpi} dpi")
    elif por_reconocer:
        print(f"\nTesseract no está disponible: {detalle}")
        print(f"{len(por_reconocer)} documentos se quedarán sin reconocer.")
        print("Se aplica lo que hay en caché y el pipeline continúa.\n")

    hechos, desde_cache, fallidos = 0, 0, []
    sin_reconocer: list[str] = []
    avisos_totales: list[tuple[str, list[str]]] = []

    for doc_id in tqdm(doc_ids, desc="ocr", unit="doc"):
        destino = config.TEXTO_CRUDO / f"{doc_id}.json"
        if not destino.exists():
            fallidos.append((doc_id, "sin JSON de extracción"))
            continue

        registro = json.loads(destino.read_text(encoding="utf-8"))

        if doc_id in cache and not args.forzar:
            entrada = cache[doc_id]
            aplicar(destino, registro, entrada["texto"], entrada["senales"])
            desde_cache += 1
            continue

        if not hay_tesseract:
            sin_reconocer.append(doc_id)
            continue

        ruta = config.CORPUS / registro["ruta_rel"]
        if not ruta.exists():
            fallidos.append((doc_id, f"no existe {registro['ruta_rel']}"))
            continue

        resultado = ocr.procesar(ruta, max_paginas=args.paginas, dpi=args.dpi)
        if resultado.error:
            fallidos.append((doc_id, resultado.error))
            continue

        s = {"motor": ocr.MOTOR, **senales(resultado)}
        avisos = sospechoso(s)
        if avisos:
            avisos_totales.append((doc_id, avisos))

        aplicar(destino, registro, resultado.texto, s)
        cache[doc_id] = {"doc_id": doc_id, "texto": resultado.texto, "senales": s}
        # Se persiste en cada documento: un apagón a mitad del lote no debe
        # costar las horas ya invertidas.
        escribir_cache(cache)
        hechos += 1

    print(f"\nreconocidos {hechos}, desde caché {desde_cache}, fallidos {len(fallidos)}")
    if hechos:
        print(f"caché actualizada: {config.OCR_CACHE} ({len(cache)} documentos)")

    if sin_reconocer:
        print(f"\n{len(sin_reconocer)} sin reconocer, por no haber Tesseract:")
        for doc_id in sin_reconocer[:15]:
            print(f"  {doc_id}")
        print("\nEntrarán al índice vacíos. Para recuperarlos:")
        print("  winget install UB-Mannheim.TesseractOCR   (con el paquete 'spa')")
        print("  uv run python scripts/etapas/02_ocr.py")

    if avisos_totales:
        print(f"\n{len(avisos_totales)} documentos con señales sospechosas:")
        for doc_id, avisos in avisos_totales[:20]:
            print(f"  {doc_id}: {'; '.join(avisos)}")
        print("\nRevísalos contra el PDF original antes de indexarlos: el texto")
        print("inventado o ilegible entra al índice como evidencia falsa.")

    if fallidos:
        print(f"\n{len(fallidos)} fallos:")
        for doc_id, err in fallidos[:15]:
            print(f"  {doc_id}: {err[:90]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
