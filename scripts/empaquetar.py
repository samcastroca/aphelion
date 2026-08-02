"""Completa el directorio `entrega/` con la estructura que exige la §1.4.

    entrega/
      generador.py         código fuente, versionado, ya está aquí
      resultados.jsonl     lo genera este script ejecutando generador.py
      informe_tecnico.pdf  lo exporta este script desde docs/
      base_vectorial/      lo copia este script desde los índices construidos
        encoder_<nombre>/
          index.faiss
          metadata.jsonl

`generador.py` no se copia desde ningún sitio: vive aquí y hay una sola copia,
porque la §1.4 exige exactamente esta ubicación. Es autónomo —no importa
`aphelion`— porque el jurado recibe solo este directorio.
`scripts/verificar_generador.py` comprueba que sigue produciendo lo mismo que el
paquete.

Uso:
    uv run python scripts/empaquetar.py
    uv run python scripts/empaquetar.py --sin-generar   # no recalcula resultados
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from aphelion import config, vectores

RAIZ = Path(__file__).resolve().parents[1]
GENERADOR = config.ENTREGA / "generador.py"
INFORME_MD = config.DOCS / "informe_tecnico.md"
INFORME_PDF = config.ENTREGA / "informe_tecnico.pdf"


def copiar_indices(destino: Path) -> list[str]:
    """Copia cada índice ya construido, validando la alineación antes de mover
    nada: un índice desalineado no debe llegar a la entrega."""
    origen = config.BASE_VECTORIAL
    nombres = vectores.encoders_disponibles(origen)
    if not nombres:
        raise FileNotFoundError(
            f"no hay índices en {origen}; ejecuta antes scripts/03_indexar.py"
        )

    for nombre in nombres:
        indice = vectores.cargar(nombre, origen)  # valida índice vs metadata
        carpeta_origen = vectores.carpeta_encoder(nombre, origen)
        carpeta_destino = destino / f"encoder_{nombre}"
        if carpeta_origen.resolve() == carpeta_destino.resolve():
            print(f"  encoder_{nombre}: {len(indice):,} fragmentos (ya en su sitio)")
            continue
        carpeta_destino.mkdir(parents=True, exist_ok=True)
        for archivo in ("index.faiss", "metadata.jsonl"):
            shutil.copy2(carpeta_origen / archivo, carpeta_destino / archivo)
        print(f"  encoder_{nombre}: {len(indice):,} fragmentos")

    return nombres


def exportar_informe() -> bool:
    """Convierte el informe a PDF con pandoc si está disponible.

    La §1.4 exige PDF de máximo 8 páginas. Sin pandoc el script no falla: avisa,
    porque la conversión puede hacerse a mano y el resto de la entrega sí queda
    armado.
    """
    if not INFORME_MD.exists():
        print(f"  aviso: no existe {INFORME_MD}")
        return False

    if shutil.which("pandoc") is None:
        print("  aviso: pandoc no está instalado; exporta el informe a PDF a mano")
        print(f"         fuente: {INFORME_MD}")
        return False

    try:
        subprocess.run(
            [
                "pandoc",
                str(INFORME_MD),
                "-o",
                str(INFORME_PDF),
                "-V",
                "geometry:margin=2.5cm",
                "-V",
                "fontsize=10pt",
                "-V",
                "lang=es",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  aviso: pandoc falló: {e.stderr.decode(errors='replace')[:200]}")
        return False

    print(f"  informe_tecnico.pdf desde {INFORME_MD.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sin-generar",
        action="store_true",
        help="no reejecuta generador.py; usa el resultados.jsonl existente",
    )
    args = ap.parse_args()

    destino = config.ENTREGA
    destino.mkdir(parents=True, exist_ok=True)

    print("índices:")
    copiar_indices(destino / "base_vectorial")

    print("\ngenerador:")
    if not GENERADOR.exists():
        print(f"  falta {GENERADOR}: es código fuente versionado, no un artefacto")
        return 1
    print(f"  generador.py ({GENERADOR.stat().st_size:,} bytes, autónomo)")

    print("\ninforme:")
    exportar_informe()

    print("\nresultados:")
    if args.sin_generar:
        if not config.RESULTADOS.exists():
            print(f"  falta {config.RESULTADOS} y se pidió --sin-generar")
            return 1
        print(f"  {config.RESULTADOS.name} sin regenerar")
    else:
        # Se ejecuta desde `entrega/` y no desde la raíz: así lo que se valida es
        # el entregable resolviendo sus rutas como lo hará el jurado.
        codigo = subprocess.call(
            [sys.executable, str(GENERADOR), "--queries", str(config.PREGUNTAS_PDF)],
            cwd=destino,
        )
        if codigo != 0:
            print("\ngenerador.py falló; la entrega no está completa")
            return codigo

    print(f"\n-> {destino}")
    faltan = [
        nombre
        for nombre in ("resultados.jsonl", "generador.py", "informe_tecnico.pdf")
        if not (destino / nombre).exists()
    ]
    if faltan:
        print(f"pendiente: {', '.join(faltan)}")
        return 1

    print("entrega completa según la §1.4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
