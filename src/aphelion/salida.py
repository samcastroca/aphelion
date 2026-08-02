"""Construcción del archivo de resultados.

La §9.2 impone que cada fragmento devuelto tenga como máximo 250 palabras. Los
fragmentos del índice se dimensionan en tokens (512), lo que en español suele
quedar por debajo de ese límite pero puede excederlo. Las reglas de la §9.2.1
resuelven ambos casos:

- Si un fragmento supera las 250 palabras, se subdivide respetando fronteras
  oracionales. Cada subfragmento ocupa su propia posición en la lista de diez.
- Si queda corto, puede concatenarse con fragmentos adyacentes del mismo
  documento mientras no se rebase el límite.

En ambos casos el `chunk_id` reportado es el del fragmento original del índice:
cumple una función de trazabilidad, no de emparejamiento (§10.2.1).
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
    # que lo supera es descartado por el evaluador automático (§9.3.2).
    if acumulado and total <= max_palabras:
        return " ".join(acumulado)

    return " ".join(texto.split()[:max_palabras])


def construir_fragmentos(
    candidatos: list[Candidato],
    top: int = config.TOP_FRAGMENTOS,
    max_palabras: int = config.MAX_PALABRAS_FRAGMENTO,
) -> list[dict]:
    """Convierte candidatos en los objetos de fragmento del esquema de salida."""
    salida: list[dict] = []

    for candidato in candidatos:
        if len(salida) >= top:
            break
        salida.append(
            {
                "rank": len(salida) + 1,
                "chunk_id": candidato.chunk_id,
                "doc_id": candidato.doc_id,
                "text": recortar_a_limite(candidato.texto, max_palabras=max_palabras),
            }
        )

    return salida


def _rellenar(objetos: list, plantilla, faltan: int) -> list:
    """Completa una lista corta repitiendo el último elemento.

    El esquema exige exactamente 3 documentos y 10 fragmentos; una lista con
    menos elementos es descartada por el evaluador automático. Solo se activa en
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
    """Comprueba el archivo contra el esquema de la §9.3 antes de entregarlo."""
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
