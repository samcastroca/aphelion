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


def _sanear_buffers(modelo) -> None:
    """Rellena los buffers no persistentes que transformers deja sin inicializar.

    Un modelo que trae su implementación en el repositorio del modelo puede
    calcular buffers en `__init__` y declararlos `persistent=False`, como hace
    gte-multilingual-base con `position_ids` y las tres tablas de RoPE. Como no
    están en el checkpoint, transformers 5 los materializa desde el meta device
    con memoria **sin inicializar** y no vuelve a ejecutar el cálculo que los
    rellena: `cos_cached` y `sin_cached` salen a cero —RoPE anula queries y
    keys— y `position_ids` sale con enteros del orden de 1e12 con los que se
    indexa la tabla de RoPE.

    En GPU eso aborta el proceso con un `device-side assert` a mitad de la
    codificación. En CPU no aborta nada: devuelve embeddings plausibles y sin
    sentido, que es la razón por la que esto se repara y no se deja pasar.

    `position_ids` sirve de detector porque su valor correcto se conoce sin
    ambigüedad; los encoders cuyo buffer llega bien (bge-m3, la familia e5) no
    se tocan. La reparación reejecuta la propia lógica del modelo —`_init_rope`—
    en vez de reimplementar el cálculo de las tablas aquí.
    """
    import torch

    emb = getattr(modelo[0].auto_model, "embeddings", None)
    pos = getattr(emb, "position_ids", None)
    if pos is None:
        return

    dispositivo = pos.device
    esperado = torch.arange(pos.numel(), device=dispositivo).view(pos.shape)
    if torch.equal(pos, esperado):
        return

    emb.register_buffer("position_ids", esperado, persistent=False)
    if hasattr(emb, "_init_rope"):
        # Construye las tablas de RoPE en CPU, como en la carga original.
        emb._init_rope(modelo[0].auto_model.config)
        emb.to(dispositivo)

    # Sin esto la reparación silenciosa sería el mismo fallo de antes con otro
    # disfraz: si una versión futura cambia los nombres, tiene que verse aquí.
    for nombre, buffer in emb.named_buffers():
        if nombre.endswith(("cos_cached", "sin_cached")) and buffer.abs().max() == 0:
            raise RuntimeError(
                f"{nombre} sigue sin inicializar tras sanear los buffers; "
                "el encoder produciría embeddings sin sentido."
            )


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
            _sanear_buffers(self._modelo)
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
