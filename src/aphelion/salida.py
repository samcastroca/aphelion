"""Construcción del archivo de resultados.

Cada fragmento devuelto puede tener como máximo 250 palabras. Los
fragmentos del índice se dimensionan en tokens (512), lo que en español suele
quedar por debajo de ese límite pero puede excederlo. Hay dos salidas:

- Si un fragmento supera las 250 palabras, se subdivide respetando fronteras
  oracionales. Cada subfragmento ocupa su propia posición en la lista de diez.
- Si queda corto, puede concatenarse con fragmentos adyacentes del mismo
  documento mientras no se rebase el límite.

En ambos casos el `chunk_id` reportado es el del fragmento original del índice:
sirve para trazar de dónde salió el texto, no para emparejar con el ground
truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .chunking import dividir_en_oraciones
from .recuperacion import Candidato, Resultado


def contar_palabras(texto: str) -> int:
    return len(texto.split())


def recortar_a_limite(
    texto: str,
    idioma: str = "es",
    max_palabras: int = config.MAX_PALABRAS_FRAGMENTO,
) -> str:
    """Recorta a `max_palabras` sin cortar una oración por la mitad.

    Si la primera oración ya excede el límite —ocurre con tablas volcadas a
    texto— se corta por palabras, porque devolver un fragmento vacío sería peor.
    """
    if contar_palabras(texto) <= max_palabras:
        return texto

    oraciones = dividir_en_oraciones(texto, idioma)
    acumulado: list[str] = []
    total = 0

    for oracion in oraciones:
        n = contar_palabras(oracion)
        if acumulado and total + n > max_palabras:
            break
        acumulado.append(oracion)
        total += n

    # Si la primera oración ya excede el límite por sí sola no hay frontera
    # oracional utilizable, y respetar el máximo tiene prioridad: un fragmento
    # que lo supera se descarta en la evaluación automática.
    if acumulado and total <= max_palabras:
        return " ".join(acumulado)

    return " ".join(texto.split()[:max_palabras])


def subdividir(
    texto: str,
    idioma: str = "es",
    max_palabras: int = config.MAX_PALABRAS_FRAGMENTO,
) -> list[str]:
    """Parte un fragmento en piezas de como máximo `max_palabras`, sin cortar
    ninguna oración por la mitad.

    Una oración que por sí sola excede el límite —ocurre con tablas volcadas a
    texto— se corta por palabras: respetar el máximo tiene prioridad, porque un
    fragmento que lo supera se descarta en la evaluación automática.
    """
    if contar_palabras(texto) <= max_palabras:
        return [texto]

    piezas: list[str] = []
    actual: list[str] = []
    total = 0

    for oracion in dividir_en_oraciones(texto, idioma):
        n = contar_palabras(oracion)

        if n > max_palabras:
            if actual:
                piezas.append(" ".join(actual))
                actual, total = [], 0
            palabras = oracion.split()
            for i in range(0, len(palabras), max_palabras):
                piezas.append(" ".join(palabras[i : i + max_palabras]))
            continue

        if actual and total + n > max_palabras:
            piezas.append(" ".join(actual))
            actual, total = [], 0

        actual.append(oracion)
        total += n

    if actual:
        piezas.append(" ".join(actual))

    return piezas or [" ".join(texto.split()[:max_palabras])]


def construir_fragmentos(
    candidatos: list[Candidato],
    top: int = config.TOP_FRAGMENTOS,
    max_palabras: int = config.MAX_PALABRAS_FRAGMENTO,
    subdividir_largos: bool = config.SUBDIVIDIR_FRAGMENTOS,
    max_por_doc: int = config.MAX_FRAGMENTOS_POR_DOC,
) -> list[dict]:
    """Convierte candidatos en los objetos de fragmento del esquema de salida.

    Cuando se subdivide, el tope por documento se cuenta sobre **posiciones
    entregadas** y no sobre fragmentos del índice. Sin eso un solo documento
    podría ocupar seis de las diez posiciones evaluadas, que es exactamente lo
    que la diversificación existe para impedir.
    """
    salida: list[dict] = []
    por_doc: dict[str, int] = {}

    for candidato in candidatos:
        if len(salida) >= top:
            break
        if por_doc.get(candidato.doc_id, 0) >= max_por_doc:
            continue

        if subdividir_largos:
            piezas = subdividir(candidato.texto, max_palabras=max_palabras)
        else:
            piezas = [recortar_a_limite(candidato.texto, max_palabras=max_palabras)]

        for pieza in piezas:
            if len(salida) >= top or por_doc.get(candidato.doc_id, 0) >= max_por_doc:
                break
            salida.append(
                {
                    "rank": len(salida) + 1,
                    # También en las piezas se reporta el chunk_id del fragmento
                    # original: sirve para trazar, no para emparejar.
                    "chunk_id": candidato.chunk_id,
                    "doc_id": candidato.doc_id,
                    "text": pieza,
                }
            )
            por_doc[candidato.doc_id] = por_doc.get(candidato.doc_id, 0) + 1

    return salida


def _rellenar(objetos: list, plantilla, faltan: int) -> list:
    """Completa una lista corta repitiendo el último elemento.

    El esquema exige exactamente 3 documentos y 10 fragmentos; una lista con
    menos elementos se descarta en la evaluación. Solo se activa en
    consultas patológicas donde el índice devuelve muy pocos candidatos.
    """
    while faltan > 0:
        objetos.append(plantilla(len(objetos) + 1))
        faltan -= 1
    return objetos


def resultado_a_dict(
    resultado: Resultado,
    top_documentos: int = config.TOP_DOCUMENTOS,
    top_fragmentos: int = config.TOP_FRAGMENTOS,
) -> dict:
    documentos = [
        {"rank": i, "doc_id": doc_id}
        for i, doc_id in enumerate(resultado.documentos[:top_documentos], start=1)
    ]
    fragmentos = construir_fragmentos(resultado.fragmentos, top_fragmentos)

    if documentos and len(documentos) < top_documentos:
        ultimo = documentos[-1]["doc_id"]
        _rellenar(
            documentos,
            lambda rank: {"rank": rank, "doc_id": ultimo},
            top_documentos - len(documentos),
        )
    if fragmentos and len(fragmentos) < top_fragmentos:
        ultimo = fragmentos[-1]
        _rellenar(
            fragmentos,
            lambda rank: {**ultimo, "rank": rank},
            top_fragmentos - len(fragmentos),
        )

    return {
        "query_id": resultado.query_id,
        "documents": documentos,
        "fragments": fragmentos,
    }


def escribir_jsonl(resultados: list[dict], destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for resultado in resultados:
            fh.write(json.dumps(resultado, ensure_ascii=False) + "\n")
    return destino


def validar(destino: Path, n_consultas: int = 50) -> list[str]:
    """Comprueba el formato del archivo antes de entregarlo."""
    problemas: list[str] = []

    with destino.open(encoding="utf-8") as fh:
        lineas = [l for l in fh if l.strip()]

    if len(lineas) != n_consultas:
        problemas.append(f"{len(lineas)} líneas, se esperaban {n_consultas}")

    vistos: set[str] = set()
    for i, linea in enumerate(lineas, start=1):
        try:
            objeto = json.loads(linea)
        except json.JSONDecodeError as e:
            problemas.append(f"línea {i}: JSON inválido ({e})")
            continue

        query_id = objeto.get("query_id")
        if not query_id:
            problemas.append(f"línea {i}: falta query_id")
            continue
        if query_id in vistos:
            problemas.append(f"línea {i}: query_id repetido ({query_id})")
        vistos.add(query_id)

        documentos = objeto.get("documents") or []
        fragmentos = objeto.get("fragments") or []

        if len(documentos) != config.TOP_DOCUMENTOS:
            problemas.append(
                f"{query_id}: {len(documentos)} documentos, se esperaban {config.TOP_DOCUMENTOS}"
            )
        if len(fragmentos) != config.TOP_FRAGMENTOS:
            problemas.append(
                f"{query_id}: {len(fragmentos)} fragmentos, se esperaban {config.TOP_FRAGMENTOS}"
            )

        for fragmento in fragmentos:
            for campo in ("rank", "chunk_id", "doc_id", "text"):
                if campo not in fragmento:
                    problemas.append(f"{query_id}: fragmento sin campo '{campo}'")
                    break
            else:
                palabras = contar_palabras(fragmento["text"])
                if palabras > config.MAX_PALABRAS_FRAGMENTO:
                    problemas.append(
                        f"{query_id} rank {fragmento['rank']}: {palabras} palabras "
                        f"(máximo {config.MAX_PALABRAS_FRAGMENTO})"
                    )

    return problemas
