"""Construcción, persistencia y consulta del índice FAISS.

`IndexFlatIP` sobre vectores normalizados: búsqueda exacta y equivalente a
similitud coseno. Con ~56k vectores no hay razón para un índice aproximado — IVF
y HNSW cambian exactitud por una velocidad que este volumen no necesita, y aquí
la exactitud del ranking es la métrica evaluada.

FAISS almacena únicamente vectores e identificadores enteros. La correspondencia
con la metadata es posicional: **la línea n de `metadata.jsonl` describe el vector
n del índice**. El orden de inserción es, por tanto, parte del contrato.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


def carpeta_encoder(nombre: str, base: Path | None = None) -> Path:
    return (base or config.BASE_VECTORIAL) / f"encoder_{nombre}"


@dataclass
class IndiceVectorial:
    nombre: str
    faiss_index: object
    metadata: list[dict]

    def __len__(self) -> int:
        return len(self.metadata)

    def buscar(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Devuelve [(posicion_interna, similitud)] ordenado de mayor a menor."""
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        puntajes, ids = self.faiss_index.search(
            np.ascontiguousarray(vector, dtype=np.float32), k
        )
        return [
            (int(i), float(s))
            for i, s in zip(ids[0], puntajes[0])
            if i != -1  # FAISS rellena con -1 si se piden más vecinos de los que hay
        ]

    def meta(self, posicion: int) -> dict:
        return self.metadata[posicion]


def construir(vectores: np.ndarray, dim: int):
    import faiss

    if vectores.shape[1] != dim:
        raise ValueError(f"dimensión inesperada: {vectores.shape[1]} != {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectores, dtype=np.float32))
    return index


def guardar(
    nombre: str, index, fragmentos: list[dict], base: Path | None = None
) -> Path:
    """Persiste índice y metadata en la estructura que exige la entrega."""
    import faiss

    if index.ntotal != len(fragmentos):
        raise ValueError(
            f"desalineación índice/metadata: {index.ntotal} vectores, "
            f"{len(fragmentos)} fragmentos"
        )

    destino = carpeta_encoder(nombre, base)
    destino.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(destino / "index.faiss"))

    with (destino / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for fragmento in fragmentos:
            fh.write(json.dumps(fragmento, ensure_ascii=False) + "\n")

    return destino


def cargar(nombre: str, base: Path | None = None) -> IndiceVectorial:
    import faiss

    origen = carpeta_encoder(nombre, base)
    ruta_index = origen / "index.faiss"
    ruta_meta = origen / "metadata.jsonl"

    if not ruta_index.exists():
        raise FileNotFoundError(f"no existe el índice: {ruta_index}")

    index = faiss.read_index(str(ruta_index))
    with ruta_meta.open(encoding="utf-8") as fh:
        metadata = [json.loads(linea) for linea in fh if linea.strip()]

    if index.ntotal != len(metadata):
        raise ValueError(
            f"índice y metadata desalineados en {nombre}: "
            f"{index.ntotal} vectores frente a {len(metadata)} registros"
        )

    return IndiceVectorial(nombre, index, metadata)


def encoders_disponibles(base: Path | None = None) -> list[str]:
    raiz = base or config.BASE_VECTORIAL
    if not raiz.exists():
        return []
    return sorted(
        d.name.removeprefix("encoder_")
        for d in raiz.iterdir()
        if d.is_dir() and d.name.startswith("encoder_") and (d / "index.faiss").exists()
    )
