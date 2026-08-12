"""Grafo de conocimiento (§7 del reto, componente bonus).

Tres etapas, un módulo por cada una:

    entidades.py    NER multilingüe sobre los fragmentos + canonicalización
    relaciones.py   tripletas por patrones lingüísticos, con procedencia
    construccion.py el grafo NetworkX y su exportación a GraphML
    busqueda.py     el canal de recuperación de la §8.5, para el barrido

**Ninguna de las tres emplea un decoder.** La §4.2 y la §8.3 prohíben las
arquitecturas generativas en la construcción del índice y en la recuperación, y
el grafo acaba interviniendo en la segunda si se conecta. Eso descarta mREBEL y
cualquier extractor seq2seq por más cómodo que resulte: aquí el NER es un
encoder bidireccional y las relaciones salen de patrones, no de generación.

**El grafo se construye pero no se conecta.** `busqueda.ranking_por_entidades`
existe para que el barrido pueda medir el canal; `Recuperador` no lo llama y
`entrega/generador.py` no lo conoce. La razón es de riesgo, no de pereza: la §7
puntúa por construir el grafo y la §8.5 dice que el equipo «puede» combinarlo,
mientras que tocar la política de recuperación obliga a escribirla también en el
entregable y a revalidar la paridad, que es lo único eliminatorio del reto. Se
mide primero; se conecta después, y solo si el barrido lo respalda.
"""

from . import construccion, entidades, relaciones

__all__ = ["construccion", "entidades", "relaciones"]
