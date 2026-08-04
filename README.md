# Aphelion — CODEFEST AD ASTRA 2026, Etapa 1

Base de conocimiento vectorial para recuperación multilingüe sobre el corpus ADL.

- **`docs/DISENO.md`** — decisiones de arquitectura y su justificación.
- **`docs/informe_tecnico.md`** — fuente del entregable en PDF (máx. 8 páginas).

## Estructura

```
data/              insumos de ADL (el corpus de 3 GB no se versiona)
docs/              DISENO.md, informe_tecnico.md          — en español
src/aphelion/      paquete del pipeline de construcción   — en español
scripts/           pipeline por etapas y herramientas de análisis
trabajo/           artefactos intermedios (regenerables, no se versionan)
entrega/
  generador.py     el entregable — autónomo, versionado
  resultados.jsonl \
  informe_tecnico.pdf > generados por scripts/etapas/05_empaquetar.py, no se versionan
  base_vectorial/  /
```

**Hay un solo `generador.py` y vive en `entrega/`**, porque es donde la §1.4 lo
exige. Es el único archivo de esa carpeta que es código fuente; el resto son
artefactos que produce `scripts/etapas/05_empaquetar.py`.

No importa `aphelion` a propósito: el jurado recibe solo el directorio
`entrega/`, así que un script que importe `src/` no arrancaría en sus manos — y
eso basta para quedar excluido de la evaluación. Su API está en inglés, igual que
el esquema de resultados que define la Tabla 2 del reto; los comentarios van en
español, como el resto del proyecto y como el informe técnico.

El precio de esa autonomía es tener la política de recuperación escrita dos
veces. `scripts/etapas/06_verificar.py` corre ambas sobre el mismo índice y exige
salidas idénticas, para que no puedan divergir en silencio.

### El paquete

Los subpaquetes siguen el recorrido del dato: del archivo original al texto, del
texto al índice, y del índice a la respuesta.

```
src/aphelion/
  config.py            rutas, encoders e hiperparámetros
  ingesta/
    catalogo.py        catálogo canónico desde el Excel de ADL (doc_id, fuente)
    extraccion.py      extractores por formato: pdf, json, csv, xlsx, pbf, txt, imagen
    ocr.py             OCR con Tesseract para los PDFs sin capa de texto
    limpieza.py        normalización, boilerplate por frecuencia, idioma
  indice/
    chunking.py        fragmentación con corte en frontera oracional
    encoders.py        BGE-M3 y multilingual-E5-large
    onnx_dml.py        el mismo encoder sobre ONNX Runtime + DirectML
    vectores.py        índice FAISS: construcción, persistencia, búsqueda
  busqueda/
    consultas.py       lectura de las 50 consultas
    recuperacion.py    RRF, boost por fenómeno, diversificación, top2 pooling
    salida.py          resultados.jsonl y validación del esquema
  evaluacion/
    metricas.py        NDCG@10, F1@3 y el test pareado que las compara
```

### Los scripts

`etapas/` construye la entrega y lo corre `pipeline.py` en orden; `analisis/` son
herramientas que se usan a mano y no forman parte de la construcción.

```
scripts/
  pipeline.py            corre las seis etapas de principio a fin
  etapas/
    01_extraer.py        texto de los 1826 documentos
    02_ocr.py            los PDFs sin capa de texto
    03_fragmentar.py     limpieza y fragmentación
    04_indexar.py        codificación e índice, por encoder
    05_empaquetar.py     resultados, informe e índices -> entrega/
    06_verificar.py      el entregable reproduce lo que produce el paquete
  analisis/
    evaluar.py           NDCG@10 y F1@3 contra el ground truth propio
    comparar.py          evalúa y ordena varias corridas a la vez, por Borda
    comparar_encoders.py un resultados.jsonl por encoder, y el fusionado
    pool_anotacion.py    arma el pool de anotación repartido entre personas
    pool_juicios.py      el mismo pool en un solo archivo, para una pasada
    submuestra.py        elige el subconjunto con el que iterar rápido
    barrido.py           políticas de recuperación sobre el índice de la entrega
    barrido_completo.py  corre recetas, presets o una rejilla a medida
    techo.py             ¿la culpa es de no encontrar o de no ordenar?
    validar_corpus.py    repite la comparación sobre el corpus, no la submuestra
```

El catálogo de recetas —las configuraciones completas que compiten por entrarse a
la entrega— vive en `src/aphelion/evaluacion/recetas.py`, junto a las métricas con
las que se juzgan.

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
uv run python scripts/etapas/04_indexar.py --encoder bge-m3 --backend onnx --lote 8
```

La primera ejecución exporta el modelo a `trabajo/onnx/` (2,2 GB, unos 35 s) y
comprueba que sus vectores coinciden con los de PyTorch antes de codificar nada.
Medido aquí: **5,0 fragmentos/s, 19 veces el CPU**, con coseno mínimo 0,99974
frente a la referencia.

El lote pequeño no es un descuido: con DirectML el coste de copiar entre CPU y
GPU domina, y lotes de 16 o 32 salen **más lentos** que el de 8.

#### En NVIDIA

```bash
uv sync --extra cuda
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

Debe imprimir `NVIDIA GeForce RTX 5070 (12, 0)`.

Las ruedas de PyPI para Windows son solo CPU, así que la build con CUDA sale del
índice de PyTorch. El extra `cuda` lo declara en `pyproject.toml`, y eso importa:
instalar torch a mano con `uv pip install --index-url ...` no sobrevive, porque
el siguiente `uv run` sincroniza contra el lock y reinstala encima la rueda CPU.
La procedencia tiene que estar en el lock.

El canal es **cu130**, no cu128: en Blackwell (sm_120) hace falta CUDA 12.8 o
superior —las anteriores fallan con un `no kernel image is available` que
despista—, pero cu128 se quedó en torch 2.11 y este proyecto exige 2.13. cu130
tiene ruedas para Windows desde cp310 hasta cp314.

## Pipeline

En una máquina nueva, un solo comando. Instala dependencias, detecta la GPU, pone
la build de PyTorch que corresponda y construye la entrega entera:

```powershell
.\ejecutar.ps1
```

Lo único manual es copiar el corpus de ADL a `data\CORPUS CODEFEST AD ASTRA 2026\`.

Sin parámetros pregunta qué hacer, para no tener que recordar ninguna opción:

```
=== Qué quieres hacer ===
  1) Construir la entrega completa                    (lo normal)
  2) Codificar solo mi parte, repartiendo entre varias PCs
  3) Reanudar desde una etapa                         (tras un fallo)
  4) Solo preparar el entorno, sin procesar nada
  Elige [1]:
```

Elegir la 2 pregunta entre cuántas máquinas se reparte y **cómo**, porque no
todas rinden igual:

```
=== Cómo se reparte la carga ===
  1) Todas parecidas               33% / 33% / 33%
  2) Una el doble que las demás    50% / 25% / 25%
  3) Una el triple que las demás   60% / 20% / 20%
  4) Escalonadas, de más a menos   50% / 33% / 17%
  5) Otro reparto, a mano

=== Qué tramo codifica esta máquina ===
  1) 0:50                la 1ª,  50% del corpus
  2) 50:75               la 2ª,  25% del corpus
  3) 75:100              la 3ª,  25% del corpus
  4) escribir el tramo a mano
```

La máquina potente elige el tramo grande y las otras los pequeños. Nadie reparte
porcentajes a mano ni comprueba que sumen 100. Después pregunta qué encoders
indexar. Enter en todo deja lo de siempre.

Con cualquier parámetro no pregunta nada y corre directo:

```powershell
.\ejecutar.ps1 -Auto                       # todo por defecto, sin menú
.\ejecutar.ps1 -Encoders bge-m3            # un solo índice, la mitad de tiempo
.\ejecutar.ps1 -Backend torch -Lote 32     # forzar backend y lote
.\ejecutar.ps1 -Desde 04_indexar:bge-m3    # reanudar tras un fallo
.\ejecutar.ps1 -Reparto 0:50               # su tramo, en el reparto entre PCs
.\ejecutar.ps1 -SinOcr                     # sin pasar por Tesseract
.\ejecutar.ps1 -SoloEntorno                # preparar sin procesar nada
.\ejecutar.ps1 -Forzar                     # seguir pese a los avisos de entorno
```

Con `-Encoders bge-m3` se llega antes a un índice utilizable, a costa de perder
la fusión de los dos espacios vectoriales. `-Lote` solo suele hacer falta para
bajarlo si la GPU se queda sin memoria.

Si prefieres saltarte el arranque y llamar al pipeline directamente:

```bash
uv run python scripts/pipeline.py
```

Comprueba el entorno antes de empezar —corpus, inventario, consultas, Tesseract y
CUDA— y aborta si algo falta, en lugar de descubrirlo dos horas después. Si una
etapa falla, se reanuda donde quedó:

```bash
uv run python scripts/pipeline.py --desde 04_indexar:bge-m3
```

Por etapas, si prefieres control fino:

```bash
# 1. Extraer texto de los 1826 documentos (cachea por documento)
uv run python scripts/etapas/01_extraer.py

# 2. OCR de los PDFs escaneados detectados en el paso anterior
uv run python scripts/etapas/02_ocr.py

# 3. Limpiar y fragmentar
uv run python scripts/etapas/03_fragmentar.py

# 4. Codificar e indexar (uno por encoder)
uv run python scripts/etapas/04_indexar.py --encoder bge-m3
uv run python scripts/etapas/04_indexar.py --encoder me5-large

# 5. Completar entrega/ (resultados, informe e índices)
uv run python scripts/etapas/05_empaquetar.py

# 6. Comprobar que generador.py reproduce lo que produce el paquete
uv run python scripts/etapas/06_verificar.py
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
# copiar el corpus a data/CORPUS CODEFEST AD ASTRA 2026/
uv sync --extra cuda
winget install UB-Mannheim.TesseractOCR   # más el paquete de idioma spa
uv run python scripts/pipeline.py
```

De vuelta solo hace falta `entrega/` (índices + `resultados.jsonl`), por Drive o
similar. `generador.py` ya está en el repo, así que no viaja.

**Si alguien no tiene el corpus**, mándale `trabajo/fragmentos.jsonl` (285 MB) y
que arranque en la etapa de indexado. Ojo con el orden: ese archivo tiene que
generarse **después** del OCR, o irá sin los escaneados.

```bash
uv run python scripts/pipeline.py --desde 04_indexar:bge-m3
```

El OCR es trabajo de CPU y no necesita GPU: puede correrlo quien tenga el corpus
mientras el otro prepara el entorno.

Cada etapa cachea su salida, así que interrumpir y reanudar no cuesta trabajo
perdido. La codificación guarda bloques de 2048 fragmentos en
`trabajo/embeddings/<encoder>-<backend>/<huella>/`, donde `huella` identifica el
archivo de fragmentos que los originó: cambiar la fragmentación invalida la caché
en lugar de mezclar vectores de dos corridas distintas.

### Repartir la codificación entre varias máquinas

Esa caché por bloques es lo que permite partir la etapa cara. Cada máquina toma
un porcentaje del corpus, y los tramos no tienen que ser iguales: quien tenga
mejor GPU carga con más.

Lo más fácil es `.\ejecutar.ps1` y elegir la opción 2: pregunta entre cuántas
máquinas se reparte y con qué perfil de carga, y saca los tramos solo — la
potente se queda el grande. A mano es lo mismo:

```powershell
.\ejecutar.ps1 -Reparto 0:50      # en la máquina rápida, la mitad del corpus
.\ejecutar.ps1 -Reparto 50:75     # en la segunda
.\ejecutar.ps1 -Reparto 75:100    # en la tercera
```

Con `-Reparto` la máquina prepara los fragmentos si no los tiene, codifica sus
bloques y se detiene: no empaqueta, porque el índice no está completo hasta juntar
los tramos de todas.

**Cada máquina genera su propio `fragmentos.jsonl` y no hay que mover 285 MB.**
Sale idéntico en todas: el orden viene de `sorted(glob)` y de `pool.map`, que lo
conserva; la limpieza y la detección de idioma son funciones puras del texto; la
ruta que se guarda es relativa, no absoluta; y el texto de los escaneados sale de
`data/ocr.jsonl`, que viaja versionado. Lo único que las separaría es que una se
saltara el OCR. Compruébalo antes de gastar la GPU:

```bash
uv run python scripts/etapas/04_indexar.py --huella
```

Un segundo, sin cargar ningún modelo. Si la huella no coincide en las tres, paren
ahí: sus vectores no encajarían.

Al terminar, cada una manda su carpeta de `.npy` a la que arma el índice. Se
juntan todos en la misma ruta —`trabajo/embeddings/<encoder>-<backend>/<huella>/`—
y una corrida normal los encuentra cacheados y arma el índice en segundos:

```bash
uv run python scripts/pipeline.py --desde 04_indexar:bge-m3
```

Antes de codificar, esa corrida dice cuántos bloques tiene y cuáles faltan. Si
falta alguno lo codifica ella misma, que es lo correcto, pero conviene verlo
antes de que la GPU se ponga a rellenar un tramo que nunca llegó.

Cuánto se transporta: unos 258 MB de vectores por encoder (63.000 fragmentos ×
1024 dimensiones × 4 bytes), o sea ~86 MB por máquina y encoder. Nada más.

**Conviene medir antes de repartir.** Si la máquina con GPU hace su encoder en
media hora, repartir para ahorrar veinte minutos no compensa la coordinación. La
opción existe para cuando las tres son comparables, o cuando quien tiene el
corpus es la máquina lenta. Corre un bloque y mira los frag/s que imprime.

### OCR

60 PDFs no tienen capa de texto; 48 son informes escaneados de la Defensoría, que
responden las consultas q033–q050. `scripts/etapas/01_extraer.py` los detecta y los
marca con `necesita_ocr` en su JSON de `trabajo/texto/`, de donde los lee la etapa
siguiente.

Se usa **Tesseract** (`spa+eng`: no todos los escaneados están en español). Se
descartó el OCR por modelo de visión-lenguaje: la §4.2 prohíbe arquitecturas
decoder en la construcción del índice, y aunque el OCR sea preprocesamiento
(§2.1), el texto que produce termina indexado. El riesgo de exclusión no compensa
la ganancia de calidad.

**El resultado se versiona en `data/ocr.jsonl`** (1,7 MB). Es el único artefacto
del pipeline que depende de software externo y cuesta horas de CPU, así que viaja
con el repositorio: una máquina nueva no necesita Tesseract para los documentos ya
reconocidos, solo para los que falten.

### Evaluación

El ground truth oficial no es público, así que se anota uno propio sobre las 50
consultas reales. Hay dos caminos, y comparten formato: los juicios acaban en
`data/anotacion/*.csv` y se consolidan en `data/ground_truth.jsonl`.

**Reparto entre personas**, con solape para medir acuerdo (kappa de Cohen):

```bash
uv run python scripts/analisis/pool_anotacion.py --anotadores 4 --top 20
# ... el equipo rellena la columna 'relevancia' en los CSV ...
uv run python scripts/analisis/pool_anotacion.py --consolidar
```

**Una sola pasada seguida**, que es lo que conviene si anota una persona o un
asistente. El pool sale de la unión del top-15 de *cada encoder por separado*,
para no sesgar la medida hacia lo que la configuración actual ya encuentra:

```bash
uv run python scripts/analisis/comparar_encoders.py      # busca una vez y cachea
uv run python scripts/analisis/pool_juicios.py --generar
uv run python scripts/analisis/pool_juicios.py --lote 1  # imprime 5 consultas
# ... se escriben los juicios en trabajo/juicios/lote_01.jsonl ...
uv run python scripts/analisis/pool_juicios.py --consolidar
```

Cada línea de juicio es `{"query_id", "chunk_id", "relevancia"}` con
relevancia 0 (no), 1 (parcial) o 2 (relevante). El campo opcional `"doc": true`
marca el documento como relevante aunque su fragmento puntúe bajo: hace falta
porque el reto usa **dos claves de emparejamiento distintas** (§10.2.1) —los
fragmentos se juzgan por su texto y los documentos por su fuente—, y un
fragmento que solo trae el título de un artículo no es evidencia aunque su
documento sí lo sea.

Que anote un modelo es admisible: el ground truth es un instrumento de medida y
no entra al índice ni al `resultados.jsonl` que se entrega, así que la §8.3 no
lo alcanza. Lo que no se puede saltar es el control — unos juicios que nadie
contrastó miden el criterio de quien anotó. Por eso conviene que una persona
anote un subconjunto en paralelo y se mire el kappa.

**Comparar corridas.** `comparar_encoders.py` escribe un `resultados_*.jsonl`
por encoder y otro con la fusión; `comparar.py` los evalúa todos contra el
ground truth y los ordena por Conteo de Borda, con desglose por fenómeno:

```bash
uv run python scripts/analisis/comparar.py trabajo/resultados_*.jsonl entrega/resultados.jsonl
```

Los documentos se emparejan por `fuente` y no por `doc_id`, como hará el jurado.
Con `--por-texto` los fragmentos también se emparejan por solape de texto en vez
de por `chunk_id`, que es lo que exige la §10.2.1 y lo único válido si la corrida
usó otra fragmentación.

## Experimentar: submuestra y barrido

Codificar los 64.484 fragmentos del corpus cuesta horas por encoder, así que
comparar configuraciones sobre el corpus completo no cabe en el tiempo
disponible. El ciclo de mejora corre sobre una submuestra:

```powershell
.\ejecutar.ps1 -Submuestra          # elige los documentos, una vez
.\ejecutar.ps1                      # menú -> "Experimentar"
.\ejecutar.ps1 -ListarPruebas       # compara lo ya corrido
```

### Recetas: candidatas completas

Una **receta** fija *todas* las dimensiones a la vez —encoders, tokens, solape,
fusión, agregación, realce, tope por documento, profundidad y umbral— y viene con
la apuesta que hace y en qué se basa. Cada receta es **una corrida**, así que la
tabla se lee de arriba abajo sin descontar el sobreajuste de haber probado miles
de variantes, y todas se comparan contra `entrega`, que es lo que está construido
hoy.

```powershell
uv run python -m aphelion.evaluacion.recetas       # el catálogo con sus parámetros
.\ejecutar.ps1 -Recetas entrega,dos-encoders       # la comparación que decide
.\ejecutar.ps1 -Recetas todas
```

| receta | encoders | tokens/solape | fusión | agregación | apuesta |
|---|---|---|---|---|---|
| `entrega` | bge-m3 | 504 / 0,15 | — | top2 | la línea base |
| `dos-encoders` | bge-m3 + me5-large | 504 / 0,15 | RRF | max | lo que se entregaba antes |
| `convexa` | bge-m3 + me5-large | 504 / 0,15 | convexa | top2 | la magnitud que RRF tira sí importa |
| `familias` | bge-m3 + gte-base | 504 / 0,15 | RRF | top2 | fusionar parientes aporta poco |
| `filtrado` | bge-m3 + me5-large | 504 / 0,15 | RRF, umbral 0,9 | top2 | la cola floja gasta posiciones |
| `barato` | me5-base | 504 / 0,15 | — | top2 | el modelo grande no se paga |
| `sin-recorte` | bge-m3 | **345** / 0,15 | — | top2 | entregar el fragmento sin truncar |
| `granular` | bge-m3 | **256** / **0** | — | max | más precisión en el top-10 |
| `contexto` | bge-m3 | **768** / 0,15 | — | top3 | más contexto por vector |

El menú (opción *"Correr recetas completas"*) las lista con sus parámetros, su
apuesta y cuántos índices le faltan a cada una, y se eligen varias con coma. Las
tres últimas cambian la fragmentación y obligan a re-codificar la submuestra; las
seis primeras comparten el índice de la entrega.

### A medida, o abriendo una dimensión

**Un experimento prueba lo que se le pide, no todas las combinaciones.** Cada
dimensión se elige por separado y lo que no se diga toma el valor de la entrega,
así que pedir un encoder, un chunking, una fusión y una agregación es *una*
corrida. El menú pregunta cada cosa con selección múltiple —`1,3` toma la primera
y la tercera, `t` todas— y las etiquetas dicen por qué está cada opción.

Sin menú, lo mismo por parámetros:

```powershell
.\ejecutar.ps1 -Barrido -Encoders bge-m3,gte-multilingual-base -Fusiones rrf,convexa
.\ejecutar.ps1 -Barrido -Nombre solo-bge -Encoders bge-m3 -MaxFusion 1
.\ejecutar.ps1 -Barrido -Chunks 256,504,768 -Solapes 0,0.15
.\ejecutar.ps1 -Barrido -Preset fusion     # una selección ya hecha
```

Los presets abren **una** dimensión y dejan el resto fija, que es como se lee un
resultado sin confundir el efecto de una cosa con el de otra: `chunking`,
`encoders`, `fusion`, `agregacion`, `politicas`, `todo`, y `rapido` para validar
la maquinaria en minutos. Un preset sirve para *entender* una dimensión; una
receta, para *elegir* qué entregar.

**Dónde queda cada cosa.**

```
pruebas/
  fragmentos/c504-s015.jsonl        compartido entre experimentos
  indices/c504-s015/encoder_x/      compartido entre experimentos
  <nombre>/metricas.jsonl           una línea por corrida, ordenada por Borda
  <nombre>/resumen.txt              la tabla que se imprimió
  <nombre>/config.json              qué se pidió, para repetirlo
```

Los fragmentos y los índices se comparten porque son caros y su contenido queda
determinado por (chunking, encoder): dos experimentos que pidan el mismo índice
lo reutilizan en vez de duplicar 117 MB. Las métricas van por experimento, que es
lo que se compara entre ellos. Nada de esto toca `entrega/`.

**La submuestra.** `scripts/analisis/submuestra.py` escribe `data/submuestra.json`
con tres capas, en este orden de prioridad:

1. **Todo lo que el ground truth toca** —265 documentos, 24.113 fragmentos— y va
   entero. Si un documento juzgado faltara, sus juicios se volverían ceros y la
   métrica mentiría hacia abajo, que es indistinguible de una configuración peor.
   Por eso no se avisa, se impone: un documento juzgado que no esté en el índice
   se toma del texto extraído —que es de donde el barrido fragmenta—, y si no
   aparece por ninguna de las dos vías el guion falla sin escribir nada.
2. **Los negativos difíciles**: documentos que los encoders puntúan alto sin ser
   relevantes, tomados de los rankings profundos cacheados. Un subconjunto de
   material relevante más ruido aleatorio es *más fácil* que el corpus real.
3. **Relleno estratificado** por (fenómeno, formato) hasta el objetivo de
   fragmentos, para no distorsionar la mezcla de idiomas y tipos de archivo.

Los documentos no se recortan. El barrido re-fragmenta y necesita el texto
completo, y una de las cosas que compara es la agregación a documento —max frente
a suma— cuyo resultado depende de la longitud real: recortar los largos sesgaría
esa comparación a favor de la suma. Por eso el suelo del subconjunto es ese 37%
de fragmentos y el ahorro real es de unas **2,3 veces**, no de diez.

**Lo caro y lo barato.** `barrido_completo.py` los separa:

- Cambiar **encoder, tamaño de chunk o solape** obliga a re-fragmentar y
  recodificar. Se hace una vez por combinación y queda cacheado en `pruebas/`.
- Cambiar **fusión, k₀, realce, diversificación, umbral o agregación** es
  reordenar candidatos ya en memoria. Cien políticas cuestan lo que una.

De ahí que convenga empezar por el preset `chunking`, que usa el encoder más
barato del catálogo y descarta media rejilla por una fracción del coste, y solo
después probar los encoders caros sobre el chunking que haya ganado. Y de ahí que
las seis primeras recetas compartan la fragmentación de la entrega: entre ellas se
comparan reordenando lo que ya está codificado.

**Un encoder a la vez.** Al construir los rankings, el barrido carga un encoder,
hace su pasada por las 50 consultas y lo suelta antes de pedir el siguiente. Dos
modelos *large* residentes a la vez no caben en una máquina de esta clase: cargar
BGE-M3 y mE5-large juntos falla con «el archivo de paginación es demasiado
pequeño» (os error 1455). Como después solo se reordena lo cacheado, ninguno hace
falta más allá de su pasada.

**El emparejamiento por texto es obligatorio aquí.** El ground truth se anotó
sobre chunks de 504 tokens; al probar 256 o 768 sus `chunk_id` dejan de existir.
`src/aphelion/evaluacion/emparejamiento.py` decide la relevancia por solape de
n-gramas con contención en el sentido más favorable, para que un chunk grande que
contiene al juzgado y uno pequeño contenido en él emparejen los dos. El umbral
(0,55) está calibrado por las dos caras: recupera el 99,9% de los juicios tras
re-fragmentar, y sobre la fragmentación original reproduce el emparejamiento por
`chunk_id` con una diferencia de 0,005 en NDCG@10 — una décima parte del
intervalo de confianza.

**Los encoders del catálogo.** `config.ENCODERS` tiene siete; `ENCODERS_ENTREGA`
es el que se indexa para entregar —desde que el corpus completo dijo que el
segundo no se pagaba, es solo BGE-M3—. Los otros existen para el barrido, todos
arquitecturas encoder con licencia permisiva, como exigen la §4.2 y la §4.3:

| clave | modelo | dim | por qué está |
|---|---|---:|---|
| `bge-m3` | BAAI/bge-m3 | 1024 | el principal actual |
| `me5-large` | intfloat/multilingual-e5-large | 1024 | el complementario hasta que se midió que no aportaba |
| `me5-base` | intfloat/multilingual-e5-base | 768 | barato: barre chunking en minutos |
| `me5-small` | intfloat/multilingual-e5-small | 384 | el más barato del catálogo |
| `me5-large-instruct` | intfloat/multilingual-e5-large-instruct | 1024 | asimetría instruida, para el caso es→en |
| `gte-multilingual-base` | Alibaba-NLP/gte-multilingual-base | 768 | familia distinta: más consenso nuevo en la fusión |
| `labse` | sentence-transformers/LaBSE | 768 | **control**: el informe lo descarta por argumento, esto lo mide |

Quedan fuera por licencia Jina v3 (CC-BY-NC-4.0) y por arquitectura cualquier
derivado de un backbone autoregresivo —Qwen3-Embedding, e5-mistral— por más que
lideren MTEB. La primera corrida descarga el modelo que falte.

## Notas de diseño que conviene no perder

- **`fuente` es la clave de emparejamiento** con el ground truth (§10.2.1), no el
  `doc_id`. Se toma literal del inventario de ADL. Por eso mismo la agregación a
  documento agrupa por `fuente`: hay 59 nombres repetidos en 186 archivos, y dos
  `doc_id` con la misma fuente en el top-3 cuentan como un solo acierto.
- **`generador.py` debe reproducir los resultados** partiendo solo de `entrega/`.
  Si no reproduce, la entrega queda excluida de la evaluación. Es eliminatorio.
- **La política de recuperación vive en dos sitios** —`aphelion.busqueda` y
  `entrega/generador.py`— y cualquier cambio va en los dos.
  `tests/test_paridad_entregable.py` los compara sobre un índice sintético en
  cada `pytest`; `06_verificar.py` lo repite sobre el índice real.
- **Los fragmentos se presupuestan a 504 tokens, no 512** (`CHUNK_PRESUPUESTO`):
  la ventana de mE5 incluye los especiales y el prefijo `passage: ` que el
  fragmento no trae, y sin la reserva la cola se trunca en silencio.
- **Ninguna oración cruza la frontera entre fragmentos** (§3.3).
- **Ningún modelo generativo interviene** en indexación ni recuperación (§4.2, §8.3).
- El orden de `metadata.jsonl` debe coincidir con los identificadores internos de
  FAISS: la línea *n* describe el vector *n*.
