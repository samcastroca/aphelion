# Aphelion — CODEFEST AD ASTRA 2026, Etapa 1

Base de conocimiento vectorial para recuperación multilingüe sobre el corpus ADL.

- **`docs/DISENO.md`** — decisiones de arquitectura y su justificación.
- **`docs/informe_tecnico.md`** — fuente del entregable en PDF (máx. 8 páginas).

## Estructura

```
datos/             insumos de ADL (el corpus de 3 GB no se versiona)
docs/              DISENO.md, informe_tecnico.md          — en español
src/aphelion/      paquete del pipeline de construcción   — en español
scripts/           pipeline por etapas
demo/              índice reducido para probar sin el corpus
trabajo/           artefactos intermedios (regenerables, no se versionan)
entrega/
  generador.py     el entregable — en inglés, autónomo, versionado
  resultados.jsonl \
  informe_tecnico.pdf > generados por scripts/empaquetar.py, no se versionan
  base_vectorial/  /
```

**Hay un solo `generador.py` y vive en `entrega/`**, porque es donde la §1.4 lo
exige. Es el único archivo de esa carpeta que es código fuente; el resto son
artefactos que produce `scripts/empaquetar.py`.

Está en inglés y no importa `aphelion` a propósito: el jurado recibe solo el
directorio `entrega/`, así que un script que importe `src/` no arrancaría en sus
manos — y eso basta para quedar excluido de la evaluación. El resto del código es
interno de construcción y se mantiene en español.

El precio de esa autonomía es tener la política de recuperación escrita dos
veces. `scripts/verificar_generador.py` corre ambas sobre el mismo índice y exige
salidas idénticas, para que no puedan divergir en silencio.

### El paquete

```
src/aphelion/
  config.py        rutas, encoders e hiperparámetros
  catalogo.py      catálogo canónico desde el Excel de ADL (doc_id, fuente)
  extraccion.py    extractores por formato: pdf, json, csv, xlsx, pbf, txt, imagen
  ocr.py           OCR con Tesseract para los PDFs sin capa de texto
  limpieza.py      normalización, boilerplate por frecuencia, idioma
  chunking.py      fragmentación con corte en frontera oracional
  encoders.py      BGE-M3 y multilingual-E5-large
  vectores.py      índice FAISS: construcción, persistencia, búsqueda
  recuperacion.py  RRF, boost por fenómeno, diversificación, max pooling
  salida.py        resultados.jsonl y validación del esquema
  metricas.py      NDCG@10 y F1@3
```

## Puesta en marcha

```bash
uv sync
```

### La GPU no es opcional

Medido en este entorno: **0,27 fragmentos/s en un Ryzen 5 3400G**. Sobre los
~63.000 fragmentos del índice son 65 horas por encoder. Con GPU baja a horas.

#### En Radeon (RX 6650 XT y similares)

CUDA no aplica y ROCm tampoco: gfx1032 no está soportado ni en Windows ni en
WSL2, y AMD declinó admitir el `HSA_OVERRIDE_GFX_VERSION` en Windows. El camino
es **ONNX Runtime sobre DirectML**, que sí funciona.

```bash
uv sync --extra amd
uv run python scripts/03_indexar.py --encoder bge-m3 --backend onnx --lote 8
```

La primera ejecución exporta el modelo a `trabajo/onnx/` (2,2 GB, unos 35 s) y
comprueba que sus vectores coinciden con los de PyTorch antes de codificar nada.
Medido aquí: **5,0 fragmentos/s, 19 veces el CPU**, con coseno mínimo 0,99974
frente a la referencia.

El lote pequeño no es un descuido: con DirectML el coste de copiar entre CPU y
GPU domina, y lotes de 16 o 32 salen **más lentos** que el de 8.

#### En NVIDIA

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

En la RTX 5070 —Blackwell, sm_120— hacen falta wheels de CUDA 12.8 o superior:
las anteriores fallan con un error engañoso (`no kernel image is available`).
Debe imprimir `NVIDIA GeForce RTX 5070 (12, 0)`. Si la wheel que necesitas no
existe para Python 3.14, crea el entorno con 3.12 (`uv venv --python 3.12`)
antes de instalar torch.

## Pipeline

En una máquina nueva, un solo comando. Instala dependencias, detecta la GPU, pone
la build de PyTorch que corresponda y construye la entrega entera:

```powershell
.\ejecutar.ps1
```

Lo único manual es copiar el corpus de ADL a `datos\CORPUS CODEFEST AD ASTRA 2026\`.

Si prefieres saltarte el arranque y llamar al pipeline directamente:

```bash
uv run python scripts/pipeline.py
```

Comprueba el entorno antes de empezar —corpus, inventario, consultas, Tesseract y
CUDA— y aborta si algo falta, en lugar de descubrirlo dos horas después. Si una
etapa falla, se reanuda donde quedó:

```bash
uv run python scripts/pipeline.py --desde 03_indexar:bge-m3
```

Por etapas, si prefieres control fino:

```bash
# 1. Extraer texto de los 1826 documentos (cachea por documento)
uv run python scripts/01_extraer.py

# 2. OCR de los PDFs escaneados detectados en el paso anterior
uv run python scripts/01b_ocr.py

# 3. Limpiar y fragmentar
uv run python scripts/02_fragmentar.py

# 4. Codificar e indexar (uno por encoder)
uv run python scripts/03_indexar.py --encoder bge-m3
uv run python scripts/03_indexar.py --encoder me5-large

# 5. Completar entrega/ (resultados, informe e índices)
uv run python scripts/empaquetar.py
```

El OCR va **antes** de fragmentar. Al revés, los escaneados entran al índice
vacíos y las consultas q033–q050 se quedan sin evidencia.

## Repartir el trabajo con la máquina con GPU

Nada pesado viaja por git: el corpus son 3 GB, `fragmentos.jsonl` 285 MB y el
índice completo ~612 MB, muy por encima del límite de 100 MB por archivo.

**Lo más simple es que quien tenga la GPU corra todo.** El corpus se lo dio ADL a
todo el equipo, así que no hay que transferirlo:

```bash
git clone <repo> && cd aphelion
# copiar el corpus a datos/CORPUS CODEFEST AD ASTRA 2026/
uv venv --python 3.12 && uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
winget install UB-Mannheim.TesseractOCR   # más el paquete de idioma spa
uv run python scripts/pipeline.py
```

De vuelta solo hace falta `entrega/` (índices + `resultados.jsonl`), por Drive o
similar. `generador.py` ya está en el repo, así que no viaja.

**Si no tiene el corpus**, mándale `trabajo/fragmentos.jsonl` (285 MB) y que
arranque en la etapa de indexado. Ojo con el orden: ese archivo tiene que
generarse **después** del OCR, o irá sin los escaneados.

```bash
uv run python scripts/pipeline.py --desde 03_indexar:bge-m3
```

El OCR es trabajo de CPU y no necesita GPU: puede correrlo quien tenga el corpus
mientras el otro prepara el entorno.

Cada etapa cachea su salida, así que interrumpir y reanudar no cuesta trabajo
perdido. La codificación guarda bloques de 2048 fragmentos en
`trabajo/embeddings/<encoder>/<huella>/`, donde `huella` identifica el archivo de
fragmentos que los originó: cambiar la fragmentación invalida la caché en lugar
de mezclar vectores de dos corridas distintas.

### OCR

60 PDFs no tienen capa de texto; 48 son informes escaneados de la Defensoría, que
responden las consultas q033–q050. `scripts/01_extraer.py` los detecta y los lista
en `trabajo/pendientes_ocr.txt`.

Se usa **Tesseract** (`spa+eng`: no todos los escaneados están en español). Se
descartó el OCR por modelo de visión-lenguaje: la §4.2 prohíbe arquitecturas
decoder en la construcción del índice, y aunque el OCR sea preprocesamiento
(§2.1), el texto que produce termina indexado. El riesgo de exclusión no compensa
la ganancia de calidad.

**El resultado se versiona en `datos/ocr.jsonl`** (1,7 MB). Es el único artefacto
del pipeline que depende de software externo y cuesta horas de CPU, así que viaja
con el repositorio: una máquina nueva no necesita Tesseract para los documentos ya
reconocidos, solo para los que falten.

### Evaluación

El ground truth oficial no es público, así que se anota uno propio sobre las 50
consultas reales.

```bash
uv run python scripts/05_pool_anotacion.py --anotadores 4 --top 20
# ... el equipo rellena la columna 'relevancia' en los CSV ...
uv run python scripts/04_evaluar.py --detalle
```

## Notas de diseño que conviene no perder

- **`fuente` es la clave de emparejamiento** con el ground truth (§10.2.1), no el
  `doc_id`. Se toma literal del inventario de ADL.
- **`generador.py` debe reproducir los resultados** partiendo solo de `entrega/`.
  Si no reproduce, la entrega queda excluida de la evaluación. Es eliminatorio.
- **Ninguna oración cruza la frontera entre fragmentos** (§3.3).
- **Ningún modelo generativo interviene** en indexación ni recuperación (§4.2, §8.3).
- El orden de `metadata.jsonl` debe coincidir con los identificadores internos de
  FAISS: la línea *n* describe el vector *n*.
