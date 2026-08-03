"""Comprueba el reparto de bloques entre máquinas.

Repartir la codificación solo sirve si los tramos cubren todos los bloques
exactamente una vez. Un bloque que se cuela en dos tramos cuesta el doble de
GPU; uno que no cae en ninguno deja un hueco que la máquina coordinadora tiene
que rellenar sola, y es justo lo que el reparto existía para evitar.

    uv run pytest
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def cargar_indexar():
    ruta = RAIZ / "scripts" / "etapas" / "04_indexar.py"
    spec = importlib.util.spec_from_file_location("indexar", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["indexar"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


indexar = cargar_indexar()


class TestNumeroDeBloques:
    def test_el_ultimo_bloque_puede_ir_incompleto(self):
        assert indexar.numero_de_bloques(2048) == 1
        assert indexar.numero_de_bloques(2049) == 2

    def test_sin_fragmentos_no_hay_bloques(self):
        assert indexar.numero_de_bloques(0) == 0


class TestTramoDeBloques:
    @pytest.mark.parametrize(
        "reparto,esperado",
        [
            ("0:100", (0, 31)),
            ("0:50", (0, 15)),
            ("50:100", (15, 31)),
            ("0:33.5", (0, 10)),
        ],
    )
    def test_convierte_porcentajes_en_bloques(self, reparto, esperado):
        tramo = indexar.tramo_de_bloques(reparto, 31)
        assert (tramo.start, tramo.stop) == esperado

    def test_tramos_contiguos_cubren_todo_sin_solaparse(self):
        total = 31
        cubierto: list[int] = []
        for reparto in ("0:50", "50:75", "75:100"):
            cubierto += list(indexar.tramo_de_bloques(reparto, total))

        assert cubierto == list(range(total)), "los tres tramos no forman la partición"

    def test_reparto_desigual_tambien_particiona(self):
        # El caso que motiva la opción: la máquina rápida carga con más.
        total = 47
        cubierto: list[int] = []
        for reparto in ("0:60", "60:80", "80:100"):
            cubierto += list(indexar.tramo_de_bloques(reparto, total))

        assert cubierto == list(range(total))

    @pytest.mark.parametrize("reparto", ["0-50", "50", "", "a:b"])
    def test_formato_invalido_aborta(self, reparto):
        with pytest.raises(SystemExit):
            indexar.tramo_de_bloques(reparto, 31)

    @pytest.mark.parametrize("reparto", ["50:50", "70:30", "-10:50", "0:120"])
    def test_rango_imposible_aborta(self, reparto):
        with pytest.raises(SystemExit):
            indexar.tramo_de_bloques(reparto, 31)

    def test_un_tramo_estrecho_puede_quedar_vacio(self):
        # Con pocos bloques, un 1% no alcanza a ninguno. El script lo detecta y
        # avisa en vez de dar la vuelta como si hubiera hecho su parte.
        assert len(indexar.tramo_de_bloques("0:1", 31)) == 0


class EncoderFalso:
    """Devuelve un vector reconocible por fragmento, para seguirles la pista.

    El fragmento i produce el vector [i, i, i, i]: si el orden de la fusión se
    rompe, el resultado deja de ser la secuencia 0, 1, 2, ... y se ve.
    """

    def __init__(self):
        self.codificados = 0

    def codificar_pasajes(self, textos, tam_lote=32, progreso=False):
        import numpy as np

        self.codificados += len(textos)
        return np.array([[float(t)] * 4 for t in textos], dtype=np.float32)


class TestFusionDeTramos:
    def test_tres_maquinas_producen_el_mismo_indice_que_una(self, tmp_path):
        import numpy as np

        textos = [str(i) for i in range(5000)]  # 3 bloques (2048 + 2048 + 904)
        total = indexar.numero_de_bloques(len(textos))
        assert total == 3

        # Cada "máquina" escribe en su propia carpeta, como pasaría de verdad.
        repartos = ["0:34", "34:67", "67:100"]
        carpetas = []
        for reparto in repartos:
            carpeta = tmp_path / f"maquina_{reparto.replace(':', '_')}"
            tramo = indexar.tramo_de_bloques(reparto, total)
            encoder = EncoderFalso()
            indexar.codificar_bloques(encoder, textos, carpeta, 32, tramo)
            assert encoder.codificados > 0, f"el tramo {reparto} no codificó nada"
            carpetas.append(carpeta)

        # La fusión: todos los .npy a una sola carpeta.
        fusion = tmp_path / "fusion"
        fusion.mkdir()
        for carpeta in carpetas:
            for npy in carpeta.glob("*.npy"):
                (fusion / npy.name).write_bytes(npy.read_bytes())

        assert indexar.bloques_faltantes(fusion, total) == []

        # La corrida final no debería codificar nada: todo está ya en la caché.
        encoder = EncoderFalso()
        matriz = np.vstack(
            indexar.codificar_bloques(encoder, textos, fusion, 32, range(total))
        )
        assert encoder.codificados == 0, "la fusión recodificó bloques ya cacheados"
        assert matriz.shape == (5000, 4)
        assert matriz[:, 0].tolist() == [float(i) for i in range(5000)]

    def test_un_bloque_con_otro_tamano_no_pasa_desapercibido(self, tmp_path):
        import numpy as np

        textos = [str(i) for i in range(3000)]  # 2 bloques
        cache = tmp_path / "cache"
        cache.mkdir()
        # Un bloque de otra corrida, con menos filas de las que toca.
        np.save(cache / "bloque_00000.npy", np.zeros((1000, 4), dtype=np.float32))

        with pytest.raises(ValueError, match="1000 vectores"):
            indexar.codificar_bloques(EncoderFalso(), textos, cache, 32, range(2))
