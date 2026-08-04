"""Los buffers no persistentes del encoder tienen que llegar inicializados.

`gte-multilingual-base` trae su implementación en el propio repositorio del
modelo, y esa implementación calcula en `__init__` cuatro buffers declarados
`persistent=False` —`position_ids` y las tres tablas de RoPE (`inv_freq`,
`cos_cached`, `sin_cached`)—. Al no estar en el checkpoint, transformers 5 los
materializa desde el meta device con memoria **sin inicializar** y nunca vuelve
a ejecutar el cálculo que los rellena.

El resultado no es un error de carga sino un modelo silenciosamente roto:
`cos_cached`/`sin_cached` salen a cero (RoPE anula queries y keys) e
`position_ids` sale con enteros del orden de 1e12 que se usan para indexar la
tabla de RoPE. En GPU eso revienta con un `device-side assert` a mitad de la
indexación; en CPU habría producido embeddings plausibles y sin sentido, que es
el fallo peor.

Se comprueban las dos cosas: que los buffers estén sanos y que el encoder
separe un par relacionado de uno que no lo está. Lo segundo es lo que de verdad
importa —un buffer sano que no ordene sería igual de inútil— y es lo que
detectaría una corrupción distinta a la ya conocida.

    uv run pytest tests/test_buffers_encoder.py
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aphelion.indice import encoders

ENCODER = "gte-multilingual-base"


@pytest.fixture(scope="module")
def modelo():
    # En CPU: el test verifica la inicialización de los buffers, que no depende
    # del dispositivo, y así corre en cualquier máquina.
    return encoders.Encoder(ENCODER, device="cpu").modelo


class TestBuffers:
    def test_position_ids_es_un_arange(self, modelo):
        buffers = dict(modelo[0].auto_model.named_buffers())
        pos = buffers["embeddings.position_ids"].flatten()
        assert torch.equal(pos.cpu(), torch.arange(pos.numel()))

    @pytest.mark.parametrize("nombre", ["cos_cached", "sin_cached"])
    def test_tablas_de_rope_no_estan_vacias(self, modelo, nombre):
        buffers = dict(modelo[0].auto_model.named_buffers())
        tabla = buffers[f"embeddings.rotary_emb.{nombre}"]
        # Sin inicializar salen a cero enteras; inicializadas son senos y
        # cosenos, acotados en [-1, 1] y con valores distintos de cero.
        assert tabla.abs().max().item() > 0
        assert tabla.abs().max().item() <= 1.0 + 1e-3

    def test_cos_en_la_posicion_cero_vale_uno(self, modelo):
        buffers = dict(modelo[0].auto_model.named_buffers())
        cos0 = buffers["embeddings.rotary_emb.cos_cached"][0].float()
        assert torch.allclose(cos0, torch.ones_like(cos0), atol=1e-3)

    def test_inv_freq_esta_acotado(self, modelo):
        buffers = dict(modelo[0].auto_model.named_buffers())
        inv = buffers["embeddings.rotary_emb.inv_freq"]
        # Frecuencias inversas de RoPE: (0, 1]. Sin inicializar dan ~1e17.
        assert inv.min().item() > 0
        assert inv.max().item() <= 1.0


class TestOrdenSemantico:
    def test_separa_lo_relacionado_de_lo_que_no(self, modelo):
        enc = encoders.Encoder(ENCODER, device="cpu")
        enc._modelo = modelo
        v = enc.codificar_pasajes(
            [
                "El cambio climático afecta a los glaciares andinos.",
                "Los glaciares de los Andes retroceden por el calentamiento global.",
                "Receta de arroz con leche y canela.",
            ],
            tam_lote=4,
            progreso=False,
        )
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
        similares = float(v[0] @ v[1])
        distintos = float(v[0] @ v[2])
        assert similares > distintos + 0.15, (similares, distintos)
