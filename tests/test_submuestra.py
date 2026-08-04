"""La submuestra contiene el ground truth entero, o no se escribe.

Es el invariante que le da sentido: si un documento juzgado se cae del
subconjunto, sus juicios se vuelven ceros y la métrica baja sin que nada lo
indique. Una configuración peor y una submuestra incompleta producen el mismo
número, y por eso el fallo tiene que ser ruidoso.

La comprobación de verdad corre sobre el corpus real —265 documentos, 24.113
fragmentos—, pero exige el índice construido. Esta prueba monta un índice de
juguete y hace las mismas preguntas en milisegundos, para que un cambio en la
selección falle en el primer `pytest`.

    uv run pytest tests/test_submuestra.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from aphelion import config
from aphelion.evaluacion.metricas import Juicio

RAIZ = Path(__file__).resolve().parents[1]


def cargar():
    ruta = RAIZ / "scripts" / "analisis" / "submuestra.py"
    spec = importlib.util.spec_from_file_location("submuestra", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["submuestra"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


def escribir_indice(base: Path, docs: dict[str, int]) -> None:
    """Un índice de mentira: doc_id -> cuántos fragmentos tiene."""
    carpeta = base / "encoder_falso"
    carpeta.mkdir(parents=True)
    (carpeta / "index.faiss").write_bytes(b"")  # solo tiene que existir
    with (carpeta / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for doc_id, n in docs.items():
            for i in range(n):
                fh.write(json.dumps({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-chunk-{i:04d}",
                    "formato": "pdf",
                    "fenomeno": 1,
                    "idioma": "es",
                    "fuente": f"{doc_id}.pdf",
                }) + "\n")


def escribir_ground_truth(ruta: Path, juicios: list[dict]) -> None:
    with ruta.open("w", encoding="utf-8") as fh:
        for j in juicios:
            fh.write(json.dumps(j) + "\n")


def escribir_texto(raiz: Path, doc_id: str) -> None:
    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / f"{doc_id}.json").write_text(json.dumps({
        "doc_id": doc_id,
        "fuente": f"{doc_id}.pdf",
        "fenomeno": 2,
        "tipo": "PDF",
        "texto": "contenido",
    }), encoding="utf-8")


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Índice, ground truth y texto extraído de juguete, aislados del proyecto."""
    mod = cargar()
    base = tmp_path / "base_vectorial"
    texto = tmp_path / "texto"
    texto.mkdir()
    salida = tmp_path / "submuestra.json"
    gt = tmp_path / "ground_truth.jsonl"

    monkeypatch.setattr(config, "TEXTO_CRUDO", texto)
    # Sin negativos difíciles: la caché de rankings del proyecto real no describe
    # este índice de juguete.
    monkeypatch.setattr(mod, "CACHE_RANKINGS", tmp_path / "no-hay-rankings.jsonl")

    def correr(*extra):
        monkeypatch.setattr(sys, "argv", [
            "submuestra.py",
            "--base", str(base),
            "--ground-truth", str(gt),
            "--salida", str(salida),
            *extra,
        ])
        return mod.main()

    return type("Entorno", (), {
        "mod": mod, "base": base, "texto": texto,
        "salida": salida, "gt": gt, "correr": staticmethod(correr),
    })


class TestElNucleoNoEsNegociable:
    def test_todo_documento_juzgado_acaba_en_la_submuestra(self, entorno):
        escribir_indice(entorno.base, {"F1-A-001": 3, "F1-B-002": 2, "F1-C-003": 5})
        escribir_ground_truth(entorno.gt, [
            {"query_id": "q001",
             "fragmentos": {"F1-A-001-chunk-0000": 2.0, "F1-B-002-chunk-0001": 0.0},
             "documentos": ["F1-A-001"]},
        ])

        assert entorno.correr("--objetivo", "0") == 0

        elegidos = set(json.loads(entorno.salida.read_text(encoding="utf-8"))["doc_ids"])
        # F1-B-002 entra aunque su único juicio sea un cero: sin él, ese cero
        # dejaría de contar y el denominador del NDCG cambiaría.
        assert {"F1-A-001", "F1-B-002"} <= elegidos

    def test_un_juzgado_fuera_del_indice_entra_por_el_texto_extraido(self, entorno):
        """El barrido fragmenta desde trabajo/texto, no desde el índice."""
        escribir_indice(entorno.base, {"F1-A-001": 3})
        escribir_texto(entorno.texto, "F1-TARDIO-009")
        escribir_ground_truth(entorno.gt, [
            {"query_id": "q001",
             "fragmentos": {"F1-TARDIO-009-chunk-0000": 2.0},
             "documentos": ["F1-TARDIO-009"]},
        ])

        assert entorno.correr("--objetivo", "0") == 0

        elegidos = set(json.loads(entorno.salida.read_text(encoding="utf-8"))["doc_ids"])
        assert "F1-TARDIO-009" in elegidos

    def test_un_juzgado_inalcanzable_aborta_en_vez_de_avisar(self, entorno):
        escribir_indice(entorno.base, {"F1-A-001": 3})
        escribir_ground_truth(entorno.gt, [
            {"query_id": "q001",
             "fragmentos": {"F1-FANTASMA-001-chunk-0000": 2.0},
             "documentos": ["F1-FANTASMA-001"]},
        ])

        assert entorno.correr("--objetivo", "0") == 1

    def test_y_no_deja_una_submuestra_a_medias(self, entorno):
        """Lo importante no es el código de salida sino que no haya archivo:
        el barrido consume el JSON, no el resultado del guion."""
        escribir_indice(entorno.base, {"F1-A-001": 3})
        escribir_ground_truth(entorno.gt, [
            {"query_id": "q001",
             "fragmentos": {"F1-FANTASMA-001-chunk-0000": 2.0},
             "documentos": []},
        ])

        entorno.correr("--objetivo", "0")
        assert not entorno.salida.exists()

    def test_el_objetivo_de_fragmentos_no_recorta_el_nucleo(self, entorno):
        """El objetivo es una meta para el relleno, no un tope: pedir menos
        fragmentos de los que el ground truth ocupa no puede dejar fuera a nadie."""
        escribir_indice(entorno.base, {"F1-A-001": 500, "F1-B-002": 400, "F1-C-003": 1})
        escribir_ground_truth(entorno.gt, [
            {"query_id": "q001",
             "fragmentos": {"F1-A-001-chunk-0000": 2.0, "F1-B-002-chunk-0000": 1.0},
             "documentos": ["F1-A-001", "F1-B-002"]},
        ])

        assert entorno.correr("--objetivo", "10") == 0

        elegidos = set(json.loads(entorno.salida.read_text(encoding="utf-8"))["doc_ids"])
        assert {"F1-A-001", "F1-B-002"} <= elegidos


class TestCobertura:
    def test_cuenta_juicios_y_no_documentos(self):
        """Un documento con veinte juicios pesa veinte, no uno: son ellos los
        que entran en la métrica."""
        mod = cargar()
        juicios = {"q001": Juicio(
            query_id="q001",
            fragmentos={f"F1-A-001-chunk-{i:04d}": 1.0 for i in range(20)},
            documentos={"F1-A-001"},
        )}
        cubiertos, juzgados, _, _ = mod.comprobar_cobertura(juicios, {"F1-A-001"})
        assert (cubiertos, juzgados) == (20, 20)

        cubiertos, juzgados, _, _ = mod.comprobar_cobertura(juicios, set())
        assert (cubiertos, juzgados) == (0, 20)

    def test_un_relevante_fuera_se_nota_aunque_sus_fragmentos_esten(self):
        """Las dos claves del reto son distintas: un documento puede ser
        relevante sin que ninguno de sus fragmentos lo sea."""
        mod = cargar()
        juicios = {"q001": Juicio(
            query_id="q001",
            fragmentos={"F1-A-001-chunk-0000": 1.0},
            documentos={"F1-A-001", "F1-B-002"},
        )}
        _, _, dentro, total = mod.comprobar_cobertura(juicios, {"F1-A-001"})
        assert (dentro, total) == (1, 2)


class TestRecuperarDelTextoCrudo:
    def test_sin_archivo_devuelve_none(self, tmp_path):
        mod = cargar()
        assert mod.desde_texto_crudo("F1-NO-EXISTE", tmp_path) is None

    def test_toma_formato_y_fenomeno_del_documento(self, tmp_path):
        mod = cargar()
        escribir_texto(tmp_path, "F1-X-001")
        registro = mod.desde_texto_crudo("F1-X-001", tmp_path)
        assert registro["formato"] == "pdf"
        assert registro["fenomeno"] == 2
        # Nadie lo ha fragmentado todavía; contar de menos agranda el relleno,
        # que es el error inocuo de los dos.
        assert registro["fragmentos"] == 0
