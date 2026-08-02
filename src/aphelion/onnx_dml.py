"""Codificación sobre GPU Radeon vía ONNX Runtime y DirectML.

**Por qué existe.** La máquina de desarrollo tiene una Radeon RX 6650 XT (gfx1032,
RDNA2) y un Ryzen 5 3400G de cuatro núcleos. En ese CPU la codificación del corpus
rinde 0,27 fragmentos/s: 65 horas por encoder. CUDA no aplica, y ROCm no soporta
gfx1032 ni en Windows ni en WSL2 —AMD declinó explícitamente admitir el
`HSA_OVERRIDE_GFX_VERSION` en Windows—, así que el único camino a la GPU es
DirectML. Medido aquí: **5,0 fragmentos/s, unas 19 veces el CPU**.

**Por qué no se usa `optimum`.** Su resolución de dependencias exige
`transformers<5` y el proyecto está en 5.14.1. La exportación con
`torch.onnx.export` no necesita esa capa.

**El pooling va dentro del grafo.** Es la decisión que más importa: BGE-M3 usa el
token CLS y E5 promedia con la máscara de atención. Confundirlos no revienta,
degrada en silencio —el coseno contra la referencia cae de 0,999999 a 0,81—, y es
el error más reportado con estos modelos. Al hornearlo en el ONNX desde
`config.ENCODERS[...]["pooling"]`, el lado de Python no puede equivocarse.

**Fidelidad.** El índice se construye aquí, pero `generador.py` codifica las
consultas con PyTorch en la máquina del jurado. Los dos caminos deben producir
vectores intercambiables: `verificar()` lo comprueba y exige coseno > 0.999.

Uso:
    uv sync --extra amd
    uv run python scripts/03_indexar.py --encoder bge-m3 --backend onnx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

PROVEEDOR = "DmlExecutionProvider"

# Longitud fija en lugar de dinámica: los fragmentos se construyen para no
# superar CHUNK_TOKENS y una forma estática evita que DirectML replanifique el
# grafo en cada tamaño de lote.
LARGO = config.CHUNK_TOKENS


def ruta_modelo(nombre: str) -> Path:
    return config.ONNX / nombre / "modelo.onnx"


def exportar(nombre: str, forzar: bool = False) -> Path:
    """Convierte el encoder a ONNX con su pooling y normalización incluidos."""
    import torch
    from torch import nn
    from transformers import AutoModel

    destino = ruta_modelo(nombre)
    if destino.exists() and not forzar:
        return destino

    cfg = config.ENCODERS[nombre]
    modo = cfg["pooling"]

    class Codificador(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask):
            h = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
            if modo == "cls":
                v = h[:, 0]
            else:
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            return v / v.norm(dim=1, keepdim=True)

    destino.parent.mkdir(parents=True, exist_ok=True)
    base = AutoModel.from_pretrained(cfg["modelo"], dtype=torch.float32).eval()

    ids = torch.ones(1, LARGO, dtype=torch.int64)
    mascara = torch.ones(1, LARGO, dtype=torch.int64)

    with torch.no_grad():
        torch.onnx.export(
            Codificador(base).eval(),
            (ids, mascara),
            str(destino),
            input_names=["input_ids", "attention_mask"],
            output_names=["embedding"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )

    return destino


class EncoderONNX:
    """Misma interfaz que `encoders.Encoder`, distinta maquinaria debajo."""

    def __init__(self, nombre: str, proveedor: str = PROVEEDOR):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if proveedor not in ort.get_available_providers():
            raise RuntimeError(
                f"{proveedor} no disponible. Instala el extra: uv sync --extra amd. "
                f"Disponibles: {ort.get_available_providers()}"
            )

        self.nombre = nombre
        self.cfg = config.ENCODERS[nombre]
        opciones = ort.SessionOptions()
        opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sesion = ort.InferenceSession(
            str(exportar(nombre)), opciones, providers=[proveedor, "CPUExecutionProvider"]
        )
        self.device = self.sesion.get_providers()[0]
        self.tok = AutoTokenizer.from_pretrained(self.cfg["modelo"])

    @property
    def dim(self) -> int:
        return self.cfg["dim"]

    def _codificar(self, textos: list[str], prefijo: str, tam_lote: int) -> np.ndarray:
        if prefijo:
            textos = [prefijo + t for t in textos]

        bloques = []
        for i in range(0, len(textos), tam_lote):
            e = self.tok(
                textos[i : i + tam_lote],
                padding="max_length",
                truncation=True,
                max_length=LARGO,
                return_tensors="np",
            )
            bloques.append(
                self.sesion.run(
                    ["embedding"],
                    {
                        "input_ids": e["input_ids"].astype(np.int64),
                        "attention_mask": e["attention_mask"].astype(np.int64),
                    },
                )[0]
            )
        return np.vstack(bloques).astype(np.float32)

    def codificar_pasajes(
        self, textos: list[str], tam_lote: int = 8, progreso: bool = True
    ) -> np.ndarray:
        return self._codificar(textos, self.cfg["prefijo_pasaje"], tam_lote)

    def codificar_consultas(
        self, textos: list[str], tam_lote: int = 8, progreso: bool = False
    ) -> np.ndarray:
        return self._codificar(textos, self.cfg["prefijo_consulta"], tam_lote)


def referencia_torch(nombre: str, textos: list[str]) -> np.ndarray:
    """Codifica con PyTorch y libera el modelo antes de devolver.

    El orden importa y no es un detalle de estilo. Este encoder ocupa ~2,2 GB, y
    `encoders.cargar` está memoizado: si la referencia se calcula con la sesión
    ONNX ya abierta, conviven dos copias del modelo y la máquina de desarrollo
    —16 GB, con el OCR corriendo— muere por segmentation fault. Primero PyTorch,
    se libera, y solo después se abre DirectML.
    """
    import gc

    from . import encoders

    vectores = encoders.cargar(nombre).codificar_pasajes(
        textos, tam_lote=4, progreso=False
    )
    encoders.cargar.cache_clear()
    gc.collect()
    return vectores


def verificar(
    referencia: np.ndarray,
    encoder: EncoderONNX,
    textos: list[str],
    umbral: float = 0.999,
) -> tuple[bool, float]:
    """Contrasta la salida de DirectML contra la referencia de PyTorch.

    Si los dos caminos divergen, el índice y las consultas del jurado quedan en
    espacios distintos y la recuperación se degrada sin que nada falle.
    """
    obtenido = encoder.codificar_pasajes(textos, tam_lote=4)
    cosenos = (referencia * obtenido).sum(axis=1)
    return bool(cosenos.min() >= umbral), float(cosenos.min())
