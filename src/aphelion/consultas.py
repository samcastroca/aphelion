"""Lectura del conjunto de consultas de evaluación.

El archivo oficial llega como PDF con las 50 preguntas en el formato

    q001 ¿Cómo está transformando la inteligencia artificial ...?
    q002 ¿Cómo están empleando los sistemas no tripulados ...?

donde cada pregunta puede ocupar varias líneas. El parser acumula líneas hasta
encontrar el siguiente identificador, de modo que las preguntas largas no se
truncan.

`generador.py` acepta además JSONL y texto plano, porque el formato definitivo del
archivo de consultas de la evaluación no está garantizado.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config

_ID_CONSULTA = re.compile(r"^(q\d{3})\b[\s:.\-]*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class Consulta:
    query_id: str
    texto: str

    @property
    def fenomeno(self) -> int | None:
        return config.fenomeno_de_consulta(self.query_id)


def _desde_texto(contenido: str) -> list[Consulta]:
    consultas: list[Consulta] = []
    actual_id: str | None = None
    piezas: list[str] = []

    def cerrar():
        if actual_id and piezas:
            texto = " ".join(" ".join(piezas).split())
            consultas.append(Consulta(actual_id.lower(), texto))

    for linea in contenido.splitlines():
        linea = linea.strip()
        if not linea:
            continue
        coincidencia = _ID_CONSULTA.match(linea)
        if coincidencia:
            cerrar()
            actual_id = coincidencia.group(1)
            piezas = [coincidencia.group(2)] if coincidencia.group(2) else []
        elif actual_id:
            piezas.append(linea)

    cerrar()
    return consultas


def _desde_pdf(ruta: Path) -> list[Consulta]:
    import pymupdf

    with pymupdf.open(ruta) as doc:
        contenido = "\n".join(p.get_text("text", sort=True) for p in doc)
    return _desde_texto(contenido)


def _desde_jsonl(ruta: Path) -> list[Consulta]:
    consultas: list[Consulta] = []
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if not linea.strip():
                continue
            d = json.loads(linea)
            texto = d.get("texto") or d.get("query") or d.get("pregunta") or ""
            consultas.append(Consulta(str(d["query_id"]).lower(), texto.strip()))
    return consultas


def cargar(ruta: Path | None = None) -> list[Consulta]:
    """Carga las consultas desde PDF, JSONL o texto plano."""
    ruta = Path(ruta) if ruta else config.PREGUNTAS_PDF

    if not ruta.exists():
        raise FileNotFoundError(f"no se encuentra el archivo de consultas: {ruta}")

    sufijo = ruta.suffix.lower()
    if sufijo == ".pdf":
        consultas = _desde_pdf(ruta)
    elif sufijo in (".jsonl", ".json"):
        consultas = _desde_jsonl(ruta)
    else:
        consultas = _desde_texto(ruta.read_text(encoding="utf-8"))

    consultas.sort(key=lambda c: c.query_id)
    return consultas


def guardar_jsonl(consultas: list[Consulta], destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        for c in consultas:
            fh.write(
                json.dumps(
                    {"query_id": c.query_id, "texto": c.texto, "fenomeno": c.fenomeno},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return destino
