"""Codificación semántica.

Los vectores se normalizan a norma unitaria en la codificación, de modo que el
producto interno del índice equivale a la similitud coseno.

E5 exige prefijos asimétricos (`query: ` / `passage: `): fueron parte de su
entrenamiento y omitirlos degrada la recuperación de forma notable. BGE-M3 no los
usa. La asimetría se resuelve aquí y no en los llamadores.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import config


def dispositivo_por_defecto() -> str:
    """cuda si está disponible, si no cpu. La indexación es agnóstica al equipo."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Encoder:
    def __init__(self, nombre: str, device: str | None = None):
        if nombre not in config.ENCODERS:
            raise KeyError(f"encoder desconocido: {nombre}")
        self.nombre = nombre
        self.cfg = config.ENCODERS[nombre]
        self.device = device or dispositivo_por_defecto()
        self._modelo = None

    @property
    def modelo(self):
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer

            # Algunos encoders traen su implementación en el propio repositorio
            # del modelo en vez de en transformers, y hay que autorizarla. Se
            # declara por encoder en `config.ENCODERS` y no se activa por defecto:
            # cargar código arbitrario de un repositorio remoto es una decisión
            # explícita, no un valor por omisión.
            extra = {}
            if self.cfg.get("trust_remote_code"):
                extra["trust_remote_code"] = True

            self._modelo = SentenceTransformer(
                self.cfg["modelo"], device=self.device, **extra
            )
            # El límite del modelo manda sobre el presupuesto de chunking.
            self._modelo.max_seq_length = min(
                self.cfg["max_tokens"], self._modelo.max_seq_length
            )
        return self._modelo

    @property
    def dim(self) -> int:
        return self.cfg["dim"]

    def _codificar(
        self, textos: list[str], prefijo: str, tam_lote: int, progreso: bool
    ) -> np.ndarray:
        if prefijo:
            textos = [prefijo + t for t in textos]
        vectores = self.modelo.encode(
            textos,
            batch_size=tam_lote,
            show_progress_bar=progreso,
            convert_to_numpy=True,
            normalize_embeddings=True,  # requisito para usar IndexFlatIP como coseno
        )
        return np.asarray(vectores, dtype=np.float32)

    def codificar_pasajes(
        self, textos: list[str], tam_lote: int = 32, progreso: bool = True
    ) -> np.ndarray:
        return self._codificar(textos, self.cfg["prefijo_pasaje"], tam_lote, progreso)

    def codificar_consultas(
        self, textos: list[str], tam_lote: int = 32, progreso: bool = False
    ) -> np.ndarray:
        return self._codificar(textos, self.cfg["prefijo_consulta"], tam_lote, progreso)


@lru_cache(maxsize=4)
def cargar(nombre: str, device: str | None = None) -> Encoder:
    return Encoder(nombre, device)
