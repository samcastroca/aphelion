# Aphelion — CODEFEST AD ASTRA 2026, Etapa 1

Base de conocimiento vectorial para recuperación multilingüe sobre el corpus ADL.

- **`DISENO.md`** — decisiones de arquitectura y su justificación. Es la base del informe técnico.
- **`GLOSARIO.md`** — vocabulario común del reto, para todo el equipo.

## Puesta en marcha

El proyecto está fijado a **Python 3.12**: PyTorch todavía no soporta 3.14.

```bash
uv sync
```

### Si tienes la RTX 5070

Es arquitectura Blackwell (sm_120). Las builds de PyTorch anteriores a CUDA 12.8
no la soportan y fallan con un error engañoso (`no kernel image is available`).
Instala explícitamente:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

Debe imprimir `NVIDIA GeForce RTX 5070 (12, 0)`.

## Pipeline

```bash
# 1. Extraer texto de los 1826 documentos (cachea por documento)
uv run python scripts/01_extraer.py

# 2. Limpiar y fragmentar
uv run python scripts/02_fragmentar.py

# 3. Codificar e indexar (uno por encoder)
uv run python scripts/03_indexar.py --encoder bge-m3
uv run python scripts/03_indexar.py --encoder me5-large

# 4. Generar el archivo de resultados
uv run python generador.py
```

Cada etapa cachea su salida, así que interrumpir y reanudar no cuesta trabajo
perdido. La codificación guarda bloques de 2048 fragmentos en `trabajo/embeddings/`.

### OCR

53 PDFs no tienen capa de texto; 48 son informes escaneados de la Defensoría, que
responden las consultas q033–q050. `scripts/01_extraer.py` los detecta y los lista
en `trabajo/pendientes_ocr.txt`.

Antes de procesarlos, decide el motor con evidencia:

```bash
uv run python scripts/probar_ocr.py --motores tesseract,unlimited --paginas 2
```

El criterio que manda es la **ausencia de alucinación**, no la velocidad: un motor
que inventa texto mete evidencia falsa en el índice.

### Evaluación

El ground truth oficial no es público, así que se anota uno propio sobre las 50
consultas reales.

```bash
uv run python scripts/05_pool_anotacion.py --anotadores 4 --top 20
# ... el equipo rellena la columna 'relevancia' en los CSV ...
uv run python scripts/04_evaluar.py --detalle
```

## Estructura

```
src/aphelion/
  config.py        rutas, encoders e hiperparámetros
  catalogo.py      catálogo canónico desde el Excel de ADL (doc_id, fuente)
  extraccion.py    extractores por formato: pdf, json, csv, xlsx, pbf, txt, imagen
  ocr.py           backends de OCR: tesseract y baidu/Unlimited-OCR
  limpieza.py      normalización, boilerplate por frecuencia, idioma
  chunking.py      fragmentación con corte en frontera oracional
  encoders.py      BGE-M3 y multilingual-E5-large
  vectores.py      índice FAISS: construcción, persistencia, búsqueda
  recuperacion.py  RRF, boost por fenómeno, diversificación, max pooling
  salida.py        resultados.jsonl y validación del esquema
  metricas.py      NDCG@10 y F1@3
scripts/           pipeline por etapas
generador.py       entregable: reproduce resultados.jsonl desde el índice
```

## Notas de diseño que conviene no perder

- **`fuente` es la clave de emparejamiento** con el ground truth (§10.2.1), no el
  `doc_id`. Se toma literal del inventario de ADL.
- **`generador.py` debe reproducir los resultados.** Si no reproduce, la entrega
  queda excluida de la evaluación. Es eliminatorio, no una penalización.
- **Ninguna oración cruza la frontera entre fragmentos** (§3.3).
- **Ningún modelo generativo interviene** en indexación ni recuperación (§4.2, §8.3).
- El orden de `metadata.jsonl` debe coincidir con los identificadores internos de
  FAISS: la línea *n* describe el vector *n*.
