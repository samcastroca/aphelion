"""Corre el pipeline completo de principio a fin, en una sola orden.

Pensado para la máquina con GPU: quien la tenga clona el repositorio, pone el
corpus en `data/` y lanza esto. Cada etapa cachea su salida, así que reanudar
tras una interrupción no repite trabajo hecho.

    01_extraer     texto de los 1826 documentos       -> trabajo/texto/
    02_ocr         los PDFs sin capa de texto         -> trabajo/texto/ (reescribe)
    03_fragmentar  limpieza y fragmentación           -> trabajo/fragmentos.jsonl
    04_indexar     codificación e índice, por encoder -> entrega/base_vectorial/
    05_empaquetar  resultados, informe e índices      -> entrega/
    06_verificar   el entregable reproduce lo que produce el paquete

El orden no es negociable: el OCR tiene que correr **antes** de fragmentar, o los
documentos escaneados entran al índice vacíos.

Con `--reparto A:B` esta máquina codifica solo el tramo A%-B% de los bloques y
se detiene ahí: no extrae, no fragmenta y no empaqueta. Sirve para repartir la
etapa cara entre varias máquinas con capacidad distinta, y exige que
`trabajo/fragmentos.jsonl` llegue copiado de la coordinadora.

Uso:
    uv run python scripts/pipeline.py
    uv run python scripts/pipeline.py --desde 04_indexar    # reanudar
    uv run python scripts/pipeline.py --sin-ocr             # sin Tesseract instalado
    uv run python scripts/pipeline.py --reparto 40:70       # solo su tramo
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from aphelion import config

RAIZ = Path(__file__).resolve().parents[1]
ETAPAS = Path(__file__).resolve().parent / "etapas"


def detectar_backend() -> tuple[str, str]:
    """Elige el backend de codificación según el hardware presente.

    Devuelve (backend, motivo). El orden no es negociable: CUDA es varios órdenes
    de magnitud mejor que las alternativas, DirectML es el único acceso a las
    Radeon en Windows, y la CPU es el último recurso.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "torch", f"CUDA: {torch.cuda.get_device_name(0)}"
    except ImportError:
        return "torch", "CPU — torch no está instalado"

    try:
        import onnxruntime as ort

        if "DmlExecutionProvider" in ort.get_available_providers():
            return "onnx", "DirectML (GPU Radeon)"
    except ImportError:
        pass

    return "torch", "CPU — sin GPU utilizable"


def etapas(
    encoders: list[str],
    sin_ocr: bool,
    backend: str,
    lote: int | None,
    reparto: str | None = None,
) -> list[tuple[str, list[str]]]:
    plan: list[tuple[str, list[str]]] = []

    # Con reparto, la preparación solo se corre si no hay fragmentos todavía. Si
    # el archivo ya está —generado aquí o copiado de otra máquina— se respeta:
    # regenerarlo encima cambiaría los bloques bajo los vectores ya codificados.
    #
    # Que cada máquina lo genere es seguro porque la fragmentación es
    # determinista: el orden sale de `sorted(glob)` y de `pool.map`, que lo
    # conserva; la limpieza y el idioma son funciones puras del texto; `ruta` es
    # relativa, no absoluta; y el texto de los escaneados viene de `data/ocr.jsonl`,
    # versionado. Lo que sí las separaría es saltarse el OCR en una y no en otra,
    # y por eso la huella del archivo se imprime antes de codificar nada.
    if not reparto or not config.FRAGMENTOS.exists():
        plan.append(("01_extraer", [str(ETAPAS / "01_extraer.py")]))
        if not sin_ocr:
            plan.append(("02_ocr", [str(ETAPAS / "02_ocr.py")]))
        plan.append(("03_fragmentar", [str(ETAPAS / "03_fragmentar.py")]))

    for encoder in encoders:
        comando = [str(ETAPAS / "04_indexar.py"), "--encoder", encoder, "--backend", backend]
        if lote:
            comando += ["--lote", str(lote)]
        if reparto:
            comando += ["--reparto", reparto]
        plan.append((f"04_indexar:{encoder}", comando))

    # Sin el índice completo no hay nada que empaquetar ni que verificar: esta
    # máquina solo aporta su tramo de vectores.
    if reparto:
        return plan

    plan.append(("05_empaquetar", [str(ETAPAS / "05_empaquetar.py")]))
    # Última etapa y no opcional: que `generador.py` reproduzca lo que produce el
    # paquete es lo único eliminatorio del reto (§1.4). Se corre sobre el índice
    # que acaba de quedar en la entrega.
    plan.append(
        (
            "06_verificar",
            [str(ETAPAS / "06_verificar.py"), "--base", str(config.BASE_VECTORIAL)],
        )
    )
    return plan


def comprobar_entorno(
    encoders: list[str],
    sin_ocr: bool,
    backend: str,
    motivo: str,
    reparto: str | None = None,
) -> list[str]:
    """Todo lo que puede fallar tarde, comprobado temprano.

    Descubrir que falta el corpus tras dos horas de codificación es exactamente
    el fallo que este proyecto no se puede permitir.
    """
    avisos: list[str] = []

    # Con los fragmentos ya hechos, la máquina que solo codifica su tramo no
    # necesita el corpus ni Tesseract. Si no los tiene, hay que prepararlos, y
    # entonces sí: se comprueba lo mismo que en una corrida normal.
    if reparto and config.FRAGMENTOS.exists():
        for encoder in encoders:
            if encoder not in config.ENCODERS:
                avisos.append(f"encoder desconocido: {encoder}")
        if "CPU" in motivo:
            avisos.append("sin GPU utilizable: repartir la codificación no arregla eso")
        return avisos

    if not config.CORPUS.exists():
        avisos.append(f"no existe el corpus en {config.CORPUS}")
    if not config.INVENTARIO.exists():
        avisos.append(f"no existe el inventario en {config.INVENTARIO}")
    if not config.PREGUNTAS_PDF.exists():
        avisos.append(f"no existe el archivo de consultas en {config.PREGUNTAS_PDF}")

    for encoder in encoders:
        if encoder not in config.ENCODERS:
            avisos.append(f"encoder desconocido: {encoder}")

    if "CPU" in motivo:
        avisos.append(
            "sin GPU utilizable: la codificación rinde ~0,3 frag/s y tardaría "
            "días. Instala torch con CUDA, o el extra amd si la GPU es Radeon"
        )

    if backend == "onnx":
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            avisos.append("falta onnxruntime-directml: uv sync --extra amd")

    # No es un aviso que aborte —el PDF se exporta a mano y el pipeline ya no
    # muere por él— pero conviene saberlo ahora y no al final: pandoc necesita
    # además un motor de LaTeX para producir un PDF.
    if shutil.which("pandoc") is None:
        print("  nota: sin pandoc, informe_tecnico.pdf habrá que exportarlo a mano")

    if not sin_ocr:
        from aphelion.ingesta import ocr

        # Con la caché de OCR poblada, Tesseract solo hace falta para lo que
        # falte por reconocer. No se exige a una máquina que solo va a indexar.
        en_cache = 0
        if config.OCR_CACHE.exists():
            en_cache = sum(1 for l in config.OCR_CACHE.open(encoding="utf-8") if l.strip())
        ok, detalle = ocr.disponible()
        if not ok and en_cache == 0:
            avisos.append(f"OCR no disponible y sin caché: {detalle}. Usa --sin-ocr")
        elif not ok:
            print(f"  nota: Tesseract ausente, se usarán los {en_cache} de la caché")

    return avisos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default=",".join(config.ENCODERS))
    ap.add_argument("--desde", help="nombre de etapa desde la que reanudar")
    ap.add_argument("--sin-ocr", action="store_true", help="omite la etapa de OCR")
    ap.add_argument(
        "--backend",
        choices=("auto", "torch", "onnx"),
        default="auto",
        help="auto elige según el hardware presente",
    )
    ap.add_argument("--lote", type=int, help="tamaño de lote de codificación")
    ap.add_argument(
        "--reparto",
        metavar="A:B",
        help="codifica solo el tramo A%%-B%% de los bloques, para repartir entre máquinas",
    )
    ap.add_argument(
        "--forzar",
        action="store_true",
        help="sigue aunque la comprobación de entorno encuentre problemas",
    )
    args = ap.parse_args()

    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]

    backend, motivo = detectar_backend()
    if args.backend != "auto":
        backend = args.backend
        motivo += f" (backend forzado a {backend})"
    print(f"hardware: {motivo}  ->  backend {backend}")

    # El lote lo decide el hardware, no el gusto. DirectML rinde mejor con lotes
    # pequeños —el coste de copiar entre CPU y GPU domina, y 16 o 32 salen más
    # lentos que 8—, mientras que en CUDA los fragmentos de 512 tokens dejan sitio
    # de sobra para 128 y el rendimiento sube con el tamaño.
    if args.lote:
        lote = args.lote
    elif backend == "onnx":
        lote = 8
    elif motivo.startswith("CUDA"):
        lote = 128
    else:
        lote = None

    plan = etapas(encoders, args.sin_ocr, backend, lote, args.reparto)

    if args.desde:
        nombres = [n for n, _ in plan]
        if args.desde not in nombres:
            print(f"etapa desconocida: {args.desde}. Disponibles: {nombres}")
            return 1
        plan = plan[nombres.index(args.desde) :]

    print("\ncomprobando entorno ...")
    avisos = comprobar_entorno(encoders, args.sin_ocr, backend, motivo, args.reparto)
    if avisos:
        for aviso in avisos:
            print(f"  ! {aviso}")
        if not args.forzar:
            print("\nCorrige lo anterior o repite con --forzar.")
            return 1
        print("\n--forzar: se continúa de todos modos.\n")
    else:
        print("  entorno correcto\n")

    print(f"etapas: {' -> '.join(n for n, _ in plan)}\n")

    t_inicio = time.time()
    for nombre, comando in plan:
        print(f"{'=' * 60}\n{nombre}\n{'=' * 60}", flush=True)
        t0 = time.time()
        codigo = subprocess.call([sys.executable, *comando], cwd=RAIZ)
        minutos = (time.time() - t0) / 60
        if codigo != 0:
            print(f"\n{nombre} falló (código {codigo}) tras {minutos:.1f} min.")
            print(f"Corrige y reanuda con: --desde {nombre}")
            return codigo
        print(f"\n{nombre} listo en {minutos:.1f} min\n")

    print(f"pipeline completo en {(time.time() - t_inicio) / 60:.1f} min")
    print(f"-> {config.ENTREGA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
