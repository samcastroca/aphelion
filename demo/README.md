# Demo — índice reducido para probar el pipeline

Índice FAISS de **1168 fragmentos** que permite ejecutar `generador.py` sin
descargar el corpus de 3 GB, sin GPU y sin esperar la codificación completa.

Existe porque los artefactos reales no caben en GitHub: el índice completo pesa
~612 MB y `fragmentos.jsonl` 285 MB, muy por encima del límite de 100 MB por
archivo.

## Probarlo

```bash
uv sync
uv run python generador.py --base demo/base_vectorial --salida trabajo/mi_prueba.jsonl
```

Debe imprimir `formato validado: esquema correcto` y producir 50 líneas. El
resultado es byte-idéntico a `resultados_muestra.jsonl`, así que sirve para
comprobar que un cambio no rompió nada: si el archivo generado difiere y no
tocaste la recuperación, algo se rompió.

La primera ejecución descarga BGE-M3 (~2 GB) desde HuggingFace. Es lo único que
no está incluido.

## Qué contiene

| Archivo | Contenido |
|---|---|
| `base_vectorial/encoder_bge-m3/index.faiss` | 1168 vectores de 1024 dimensiones |
| `base_vectorial/encoder_bge-m3/metadata.jsonl` | metadata de esos fragmentos |
| `fragmentos_muestra.jsonl` | los mismos fragmentos antes de codificar |
| `resultados_muestra.jsonl` | salida de referencia para las 50 consultas |

## Cómo se construyó la muestra

Muestreo estratificado por fenómeno sobre **documentos completos**, no sobre
fragmentos sueltos: 180 documentos con hasta 12 fragmentos cada uno. Mantener los
documentos enteros permite que la agregación a nivel documento se comporte como
en el índice real.

Se excluyeron los cinco CSV de PubMed, que aportan el 56% de los fragmentos del
corpus completo con referencias bibliográficas biomédicas ajenas a las 50
consultas y habrían dominado la muestra.

**Esta demo no sirve para medir calidad de recuperación** — es el 0.8% del corpus.
Sirve para verificar que el pipeline corre, que el formato de salida valida y que
los resultados son reproducibles.
