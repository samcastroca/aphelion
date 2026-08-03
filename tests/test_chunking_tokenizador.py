"""El tokenizador de fragmentación no debe arrastrar torch.

Fragmentar es trabajo de CPU puro repartido entre un proceso por núcleo. Cuando
el tokenizador llega vía `transformers.AutoTokenizer`, cada worker importa torch
de forma transitiva y compromete ~1,35 GiB antes de leer un solo documento; a 15
procesos son ~20 GiB de memoria comprometida que el cargador de DLL de Windows
resuelve con `WinError 1455` (archivo de paginación demasiado pequeño) y el pool
muere entero.

El backend Rust de `tokenizers` produce los mismos ids con ~24 MiB, así que el
invariante que se fija aquí es: cargar el tokenizador no importa torch.

    uv run pytest tests/test_chunking_tokenizador.py
"""

from __future__ import annotations

import subprocess
import sys

from aphelion import config
from aphelion.indice import chunking

# Se ejecuta en un intérprete limpio: en el proceso de pytest, cualquier otro
# test podría haber importado torch antes y el aserto pasaría por accidente.
SONDA = """
import sys
from aphelion.indice import chunking
chunking._tokenizador(%r)
print("torch" in sys.modules)
"""


def test_cargar_el_tokenizador_no_importa_torch():
    salida = subprocess.run(
        [sys.executable, "-c", SONDA % config.ENCODER_PRINCIPAL],
        capture_output=True,
        text=True,
        check=True,
    )
    assert salida.stdout.strip().endswith("False"), (
        "el tokenizador arrastró torch: cada worker de 03_fragmentar pagará "
        "~1,35 GiB de memoria comprometida"
    )


def test_contar_devuelve_el_numero_de_tokens_del_encoder():
    tok = chunking._tokenizador(config.ENCODER_PRINCIPAL)
    # 22 tokens según el propio tokenizador de BGE-M3, sin tokens especiales.
    # Fija que cambiar de backend no cambia el presupuesto que se está midiendo.
    texto = "Hola mundo. Esto es una prueba de tokenización con acentos: canon, niño, acción."
    assert chunking._contar(tok, texto) == 22


def test_partir_por_tokens_respeta_el_presupuesto():
    tok = chunking._tokenizador(config.ENCODER_PRINCIPAL)
    texto = " ".join(f"palabra{i}" for i in range(400))

    trozos = chunking._partir_por_tokens(texto, tok, presupuesto=64)

    assert len(trozos) > 1
    assert all(chunking._contar(tok, t) <= 64 for t in trozos)
    # Los offsets no deben perder contenido por el camino.
    assert "palabra0" in trozos[0] and "palabra399" in trozos[-1]
