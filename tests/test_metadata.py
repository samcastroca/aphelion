"""Fija el esquema de metadata contra la Tabla 1 del reto.

La §3.4 nombra ocho campos obligatorios, literalmente, y la §1.4 exige que
`metadata.jsonl` los incluya todos. Los nombres son los que son —mezcla de
inglés y español— y renombrarlos por gusto significa que esos campos dejan de
estar. Este archivo existe para que ese cambio falle aquí y no en la evaluación.

Los campos adicionales sí son libres: la §3.4 los autoriza y los ejemplifica en
español ("idioma", "título del documento").

    uv run pytest
"""

from __future__ import annotations

import pytest

from aphelion.indice.chunking import Fragmento

# Tabla 1: Campos de metadata obligatorios por fragmento. Copiado del reto, con
# el tipo que exige cada uno: 'cadena' -> str, 'entero' -> int.
TABLA_1 = {
    "doc_id": str,
    "chunk_id": str,
    "fuente": str,
    "formato": str,
    "fenomeno": int,
    "posicion": int,
    "num_tokens": int,
    "texto": str,
}


@pytest.fixture
def fragmento():
    return Fragmento(
        doc_id="F3-ALERTAS-412",
        chunk_id="F3-ALERTAS-412-chunk-0007",
        fuente="ALERTAS_informes049.pdf",
        formato="pdf",
        fenomeno=3,
        posicion=7,
        num_tokens=498,
        texto="El texto del fragmento.",
        idioma="es",
        observatorio="Alertas_Tempranas",
        ruta="F3_Dinamicas_Territoriales/Alertas_Tempranas/informes049.pdf",
        titulo=None,
    )


class TestTabla1:
    def test_estan_los_ocho_campos_obligatorios(self, fragmento):
        faltan = TABLA_1.keys() - fragmento.to_dict().keys()
        assert not faltan, f"la Tabla 1 exige {sorted(faltan)}"

    @pytest.mark.parametrize("campo,tipo", TABLA_1.items())
    def test_cada_campo_lleva_el_tipo_que_pide_la_tabla(self, fragmento, campo, tipo):
        valor = fragmento.to_dict()[campo]
        # bool es subclase de int en Python y no serviría donde se pide 'entero'.
        assert type(valor) is tipo, f"{campo} es {type(valor).__name__}, se pide {tipo.__name__}"

    def test_la_posicion_empieza_en_cero(self):
        """'Índice ordinal del fragmento dentro del documento (empieza en 0)'."""
        primero = Fragmento(
            doc_id="D", chunk_id="D-chunk-0000", fuente="d.pdf", formato="pdf",
            fenomeno=1, posicion=0, num_tokens=10, texto="t", idioma="es",
            observatorio="O", ruta="d.pdf",
        )
        assert primero.to_dict()["posicion"] == 0

    def test_los_campos_extra_son_los_que_declaramos(self, fragmento):
        # No es un requisito del reto, que los permite sin condiciones. Es para
        # enterarse si alguien añade uno: cada campo aquí engorda metadata.jsonl
        # una vez por fragmento, y son decenas de miles.
        extra = fragmento.to_dict().keys() - TABLA_1.keys()
        assert extra == {"idioma", "observatorio", "ruta", "titulo"}
