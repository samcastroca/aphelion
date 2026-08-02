---
title: "Base de Conocimiento Vectorial — Documento Técnico"
subtitle: "CODEFEST AD ASTRA 2026 · Etapa 1"
lang: es
---

Este documento describe las decisiones de diseño de nuestra base de conocimiento
vectorial: la estrategia de chunking y por qué la elegimos, los encoders y los
criterios con que los seleccionamos, el tipo de índice FAISS que empleamos y qué
pasó con el grafo de conocimiento. Antes de eso presentamos las mediciones sobre
el corpus, porque casi todas las decisiones salieron de ahí.

## 1. Qué encontramos en el corpus

Lo primero que hicimos fue medir. Todas las cifras de este documento vienen del
inventario completo de ADL, 1826 archivos que verificamos uno a uno contra disco.

| Formato | Archivos | Texto extraído |
|---|---:|---:|
| PDF | 759 | 131,6M caracteres |
| CSV | 26 | 111,9M caracteres |
| PBF | 73 | 7,3M caracteres |
| JSON | 954 | 4,7M caracteres |
| XLSX | 4 | 1,4M caracteres |
| Imagen / TXT | 9 | OCR / texto plano |
| **Total** | **1826** | **~257M caracteres** |

Cuatro hallazgos condicionaron todo lo demás.

**La recuperación cross-lingual no es un caso borde, es el problema principal.**
Las 50 consultas están en español, pero el corpus es mayoritariamente inglés: de
los fragmentos que terminamos indexando, 130.090 están en inglés, 11.066 en
español y 7.617 en portugués. Una consulta en español tiene que recuperar
documentos en inglés la mayor parte del tiempo, y eso pesó en la elección de
encoder más que cualquier otro criterio.

**60 PDFs no traen capa de texto.** 48 de ellos son informes escaneados de
alertas tempranas de la Defensoría del Pueblo, que es justamente el material que
responde las consultas q033 a q050. Si los dejábamos fuera, el sistema quedaba
ciego en todo el tercer fenómeno.

**Los tamaños son incomparables entre sí.** Hay alertas en JSON de 1.300
caracteres y atlas de RESDAL de varios cientos de páginas. Esto nos obligó a
descartar la suma de puntuaciones para agregar a nivel de documento, y a diseñar
un chunking que aguantara los dos extremos.

**Unos pocos archivos tabulares se estaban comiendo el índice.** Los 30 archivos
CSV y XLSX aportaban 90.442 de 149.571 fragmentos, y solo tres exportaciones
bibliográficas de PubMed ocupaban el 51,5%. Son listados de referencias
biomédicas que no responden ninguna de las 50 consultas, pero consumían más de la
mitad del tiempo de codificación y competían contra la evidencia buena en cada
búsqueda.

---

## 2. Estrategia de chunking

**512 tokens, 15% de superposición, cortes en frontera oracional.**

### Qué combinamos y qué dejamos fuera

Nuestra estrategia es híbrida. Toma el presupuesto fijo de tokens, que es lo que
acota el coste de codificación y el tamaño del índice; usa las fronteras de
oración para decidir dónde puede caer el corte; y añade superposición entre
fragmentos consecutivos para que una idea no quede partida entre dos.

El presupuesto nunca parte una oración. Cuando la siguiente no cabe, el fragmento
cierra donde terminó la anterior. La superposición se implementa arrastrando
hacia adelante las últimas oraciones del fragmento previo, así que también sus
fronteras caen en límites oracionales.

Descartamos el chunking jerárquico o estructural. El corpus llega en siete
formatos y solo los PDF conservan señales de estructura fiables; aplicarlo
habría significado un criterio distinto por formato sin una ganancia clara.

También descartamos el chunking semántico puro. Rinde bien a nivel de fragmento,
pero colapsa a nivel de documento porque produce fragmentos de unos 43 tokens que
rompen el contexto, y el F1@3 se mide precisamente sobre documentos.

### Ninguna oración cruza la frontera entre fragmentos

Este requisito descarta cortar por conteo de tokens y obliga a ir acumulando
oraciones completas. Segmentamos con `pysbd`, que trae reglas para los tres
idiomas del corpus. Los tokens los cuenta el tokenizador del propio encoder y no
una aproximación por palabras, porque el presupuesto que importa es el del
modelo.

Verificamos el cumplimiento instrumentando el código y midiendo qué porcentaje de
fragmentos salió por alguna ruta que no corta en frontera oracional:

| Formato | Corte por separador débil | Ventana de tokens |
|---|---:|---:|
| PDF | 0,0% | 0,0% |
| JSON | 0,0% | 0,0% |
| XLSX | 0,0% | 0,0% |
| CSV | 15,2% | 1,5% |
| PBF | 60,4% | 0,0% |

Todo el texto en prosa cumple sin excepción. Las desviaciones se concentran donde
no hay oraciones que respetar: filas de CSV y volcados de atributos de teselas
PBF. Cuando una de esas «oraciones» excede el presupuesto del modelo, cortamos
primero por separadores débiles (`;`, `,`, `|`), que al menos preservan unidades
legibles, y solo troceamos por ventana de tokens lo que aún no quepa.

### Por qué 512

Las evaluaciones publicadas sitúan el fragmento de 512 tokens con solape alineado
a oraciones como el punto de partida razonable, y el pico de fidelidad alrededor
de 1024, así que el rango 512–1024 es donde hay que buscar.

El tamaño tiene una tensión incómoda: 512 favorece el NDCG@10 y 1024 favorece el
F1@3. Como el Conteo de Borda pondera las dos métricas por igual, no hay un
ganador a priori y dejamos el valor sujeto a un barrido sobre {384, 512, 768}
contra nuestro conjunto de evaluación. La superposición entra al mismo barrido:
hay evidencia reciente de que no aporta beneficio medible, y cuesta un 15%
adicional de vectores.

### Un tope para los archivos tabulares

Limitamos cada documento tabular a 400 fragmentos, lo que baja el índice de
149.571 a unos 63.000. Es un post-filtro sobre el campo `formato` de la metadata,
aplicado durante la indexación.

No excluimos ningún archivo. El emparejamiento a nivel de documento se hace por
el campo `fuente`, así que todos tienen que seguir siendo alcanzables. El tope
solo afecta a CSV y XLSX, donde cada fila es una unidad independiente y la fila
3.000 no aporta nada que no aportara la 400. Los PDF largos y legítimos quedan
intactos: el más extenso es una ley con 1.960 fragmentos.

### Metadata por fragmento

Emitimos los ocho campos obligatorios —`doc_id`, `chunk_id`, `fuente`, `formato`,
`fenomeno`, `posicion`, `num_tokens`, `texto`— y añadimos cuatro que nos sirven
para post-filtros: `idioma`, `observatorio`, `ruta` y `titulo`.

En `formato` ponemos el formato real del archivo (`pdf`, `json`, `csv`, `xlsx`,
`pbf`, `txt`, `imagen`). La tabla de campos obligatorios menciona solo `pdf`,
`html` o `md`, pero el corpus que recibimos tiene siete formatos y reportar el
real es lo único que hace utilizable el filtrado por metadata.

**El corpus queda en unos 63.000 fragmentos**, con mediana de 499 tokens, sobre
1758 de los 1826 documentos. Los 68 restantes son los 60 que necesitan OCR y 8
archivos JSON que solo contienen fechas.

---

## 3. Encoders y criterios de selección

Usamos dos encoders, cada uno con su índice FAISS independiente.

| Modelo | Arquitectura | Dim. | Contexto | Licencia | Rol |
|---|---|---:|---:|---|---|
| `BAAI/bge-m3` | XLM-RoBERTa (encoder) | 1024 | 8192 | MIT | Principal |
| `intfloat/multilingual-e5-large` | XLM-RoBERTa (encoder) | 1024 | 512 | MIT | Complementario |

**Soporte multilingüe.** Fue el criterio decisivo, porque las consultas están en
español y el 87% de los fragmentos en inglés. BGE-M3 se entrena explícitamente
para recuperación cross-lingual y rinde 67,8 nDCG@10 en MIRACL sobre 18 idiomas,
frente a 65,4 de mE5-large. Los dos operan de forma nativa en español, inglés y
portugués.

**Dimensionalidad.** Ambos producen vectores de 1024 dimensiones. Con unos 63.000
fragmentos el índice ocupa alrededor de 250 MB por encoder, cómodo para búsqueda
exhaustiva. No buscamos modelos de mayor dimensión porque más dimensiones no
garantizan mejor recuperación y sí encarecen todo.

**Longitud máxima de entrada.** La ventana de 8192 tokens de BGE-M3 nos deja las
manos libres en el chunking. mE5-large se queda en 512, que es justamente nuestro
presupuesto: diseñamos el chunking contra el más restrictivo de los dos para que
ningún fragmento sufra truncamiento silencioso en ninguno de los dos índices.

**Rendimiento en benchmarks.** Los dos son modelos de recuperación densa, no
optimizados para clasificación ni para similitud de pares. Las cifras de MIRACL
que citamos arriba son de recuperación, que es la tarea que nos ocupa.

**Licencia.** MIT en los dos casos.

**Eficiencia.** 568M y 560M parámetros. El índice se construye una sola vez y
codificar una consulta toma milisegundos. Sin GPU la indexación del corpus
completo tarda días, así que el pipeline es agnóstico del dispositivo y la
construcción se hace donde haya aceleración disponible.

### Por qué un segundo encoder

mE5-large mantiene un espacio vectorial más particionado por idioma, lo que
reduce la tendencia a devolver documentos en un idioma distinto al de la
consulta. Eso compensa una debilidad conocida de BGE-M3 y nos da dos rankings de
espacios distintos que podemos combinar, que es de donde viene la robustez. E5
necesita prefijos asimétricos (`query: ` y `passage: `); los aplicamos dentro del
encoder para que ningún llamador tenga que acordarse.

### Modelos que consideramos y descartamos

- **Qwen3-Embedding** lidera MTEB multilingüe, pero deriva de un backbone
  autoregresivo. El reto prohíbe las arquitecturas decoder para generar
  embeddings, así que queda fuera por más que sea el mejor del ranking.
- **Jina Embeddings v3** tiene licencia CC-BY-NC-4.0, incompatible con el
  criterio de licencia.
- **LaBSE** se entrenó para alinear pares de frases paralelas y no tiene noción
  de relevancia temática asimétrica; rinde 18,80 en recuperación zero-shot.

### Sobre las otras cabezas de BGE-M3

El modelo produce representaciones densas, sparse y multi-vector en la misma
pasada, y combinarlas sube su nDCG@10 en MIRACL de 67,8 a 70,0. Ninguna de las
tres es un decoder, así que usarlas sería legítimo.

Usamos solo la densa, y queremos ser explícitos sobre por qué. La recuperación
léxica rinde peor justamente en escenarios cross-lingual, donde el vocabulario
compartido entre consulta y documento es mínimo, y ese es nuestro caso dominante.
Esperaríamos que ayudara con las siglas y los topónimos que sí cruzan idiomas
—NBQR, ASAT, Chocó, Arauca— y que estorbara en el resto. Sin ground truth no
podemos saber cuál de los dos efectos gana, y no nos parece defendible añadir una
componente de recuperación cuyo efecto no hemos medido. Queda documentada como lo
primero que probaríamos con anotaciones en la mano.

---

## 4. Índice FAISS

Usamos **`IndexFlatIP` sobre vectores normalizados**, un índice independiente por
encoder.

Con unos 63.000 vectores, la búsqueda exhaustiva es exacta y se resuelve en
milisegundos. `IndexIVFFlat` e `IndexHNSW` cambian exactitud por velocidad, y esa
velocidad no nos hace falta a este volumen. Además, aquí la exactitud del ranking
es literalmente lo que se evalúa: perder recuperación para ganar latencia sería
un mal negocio.

Elegimos producto interno y no L2 porque, con los vectores normalizados de
antemano, el producto interno equivale a la similitud coseno, que es la medida
con la que comparamos consulta e índice.

FAISS guarda solo los vectores y sus identificadores internos, así que la
metadata va en un almacén aparte. El vínculo es posicional: la línea *n* de
`metadata.jsonl` describe el vector *n*. El orden de inserción es parte del
contrato, y verificamos la alineación tanto al guardar como al cargar, porque una
desalineación no produce ningún error visible: simplemente devuelve textos que no
corresponden.

El índice se serializa con `faiss.write_index()` y se abre con
`faiss.read_index()` sin dependencias adicionales.

---

## 5. Grafo de conocimiento

No lo implementamos. Es un componente opcional y preferimos invertir el tiempo en
la calidad de la base vectorial, que es lo que se evalúa en esta etapa. La entrega
no incluye subcarpeta `grafo/` ni archivo `grafo.graphml`, y el módulo de
recuperación no tiene ninguna rama que dependa de él.

---

## 6. Preprocesamiento

### Extracción

| Formato | Cómo lo tratamos |
|---|---|
| PDF | PyMuPDF ordenando por bloques de lectura, que importa en los diseños a dos columnas de los atlas RESDAL y los informes de CSIS |
| JSON (artículo) | `title` más `body_paragraphs` o `body_text` concatenados en orden; `url`, `date`, `authors` y `tags` van a metadata, no al cuerpo |
| JSON (catálogo) | Cada registro de la lista se emite como bloque independiente |
| CSV / XLSX | Cabecera primero y luego cada fila como unidad, en pares `columna: valor`, para que cada dato conserve el nombre de su columna |
| PBF | Recorremos capas y elementos, volcando sus atributos como pares `atributo: valor` |
| Imagen | OCR |

En los PBF deduplicamos dentro de cada tesela, no entre teselas. La recomendación
de quedarse con una sola versión de cada elemento apunta a los niveles de zoom
repetidos, pero ADL asigna un `doc_id` distinto a cada archivo PBF: si
descartáramos teselas completas volveríamos inalcanzables documentos que el
ground truth podría marcar como relevantes.

Las alertas de la Defensoría nos dieron trabajo aparte. Su contenido sustantivo
—tema de riesgo, municipios, grupos armados— vive en el objeto `alerta_meta` y no
en `body_paragraphs`. Extraer solo el cuerpo dejaba cada alerta en un párrafo
suelto sin información útil.

### OCR

Usamos Tesseract con los paquetes de español e inglés, a 200 dpi. No todos los
escaneados están en español: 48 son de la Defensoría, pero el resto son informes
de CSET y de CSIS.

Llegamos a implementar una alternativa con un modelo de visión-lenguaje
(`baidu/Unlimited-OCR`, licencia MIT, que entiende el layout) y la descartamos.
El OCR es preprocesamiento y el reto lo recomienda explícitamente, pero el texto
que produce termina dentro del índice, y no quisimos exponernos a que se leyera
como una violación de la prohibición sobre arquitecturas decoder. A eso se suma
un riesgo de fondo: un modelo generativo que alucina mete evidencia falsa en el
índice, y esa evidencia puede acabar presentada como respuesta. Tesseract, cuando
falla, produce ruido evidente.

Sobre cada documento procesado medimos tres señales, porque un OCR que falla en
silencio es peor que uno que revienta: densidad de diacríticos —solo si el texto
resulta estar en español—, proporción de caracteres no imprimibles y palabras por
página. Lo que dispare alguna señal lo revisamos contra el PDF original antes de
indexarlo.

### Limpieza

Normalizamos a UTF-8, eliminamos caracteres de control y espacios redundantes, y
detectamos el idioma predominante, que guardamos como metadata.

Los elementos repetitivos los quitamos por frecuencia de línea y no por patrones
fijos. Una línea corta que se repite cuatro o más veces dentro de un documento es
un encabezado, un pie o un rótulo de eje; una oración del cuerpo no lo es. El
criterio es estructural, así que funciona igual en los tres idiomas. Sobre un
capítulo del AI Index eliminó el 14% de las líneas sin tocar la prosa.

### Identificación de documentos

No inventamos identificadores. Replicamos el esquema del inventario de ADL:

```
doc_id = {fenomeno}-{codigo_observatorio}-{consecutivo:03d}   p. ej. F1-CSET-001
fuente = campo "Nombre estandarizado" del inventario, literal
```

Esto importa porque la comparación con el ground truth a nivel de documento se
hace por `fuente`, no por el `doc_id` que cada equipo se inventa. Cualquier
divergencia respecto al nombre original arruinaría el F1@3 por bueno que fuera el
sistema.

La ingesta recorre el inventario y nunca un listado del directorio. Así quedan
fuera por construcción los archivos que contaminarían el índice: los `.DS_Store`,
las hojas del propio inventario y, sobre todo, el PDF con las preguntas, que si
se indexa mete el examen dentro del corpus.

Una salvedad que verificamos: 59 nombres estandarizados están repetidos en
carpetas distintas, 186 archivos en total, y su contenido es genuinamente
diferente. Lo confirmamos por hash MD5, que de paso nos dijo que el corpus no
tiene duplicados exactos. Reportamos el nombre literal en `fuente` y guardamos la
ruta completa aparte para poder rastrear cuál es cuál.

---

## 7. Módulo de recuperación

```
búsqueda por índice → fusión RRF → realce por fenómeno → diversificación
                    → top-10 fragmentos → max pooling → top-3 documentos
```

La consulta se codifica con el mismo encoder que se usó al indexar, con el
prefijo que cada modelo necesita, y se normaliza igual que los vectores del
índice.

**Fusión.** Combinamos los dos rankings con Reciprocal Rank Fusion y `k₀ = 60`.
RRF opera sobre posiciones y no sobre puntuaciones, así que no le afecta que dos
encoders tengan escalas distintas. Anotamos una salvedad honesta: esa ventaja es
grande cuando se fusionan rankers heterogéneos, y aquí los dos son densos con
coseno en el mismo rango, de modo que CombSUM podría rendir igual o mejor. Es una
de las cosas que queremos comparar contra el conjunto de evaluación, junto con el
valor de `k₀`.

**Realce por fenómeno.** Sabemos a qué fenómeno corresponde cada consulta
(q001–q016 al primero, q017–q032 al segundo, q033–q050 al tercero) y el campo
`fenomeno` está en la metadata, así que podríamos filtrar en firme. No lo hacemos:
varias consultas admiten evidencia transversal. q005 y q046 tratan Colombia desde
fenómenos distintos, y q027 cruza inteligencia artificial con operaciones
espaciales. Un filtro estricto dejaría fuera documentos legítimamente relevantes,
así que aplicamos un multiplicador suave.

**Diversificación.** Limitamos cuántas de las diez posiciones puede ocupar un
mismo documento. Sin ese tope, un solo documento puede llevarse las diez; si
resulta no ser relevante, se pierde la consulta entera.

**Deduplicación.** 4.445 fragmentos del corpus, un 3%, tienen texto idéntico:
tablas reimpresas en varios informes, encabezados institucionales repetidos. Dos
copias del mismo texto tienen la misma relevancia y gastan dos posiciones para
informar una sola vez, así que las quitamos de la lista de fragmentos. No las
quitamos de la agregación a documento, porque dos documentos distintos pueden
compartir un texto y descartar uno lo volvería inalcanzable.

**Agregación a documento.** Usamos max pooling: cada documento se queda con la
puntuación de su mejor fragmento. Descartamos sumar todos sus fragmentos
recuperados por el sesgo de longitud. Un documento con cuarenta fragmentos flojos
de 0,15 acumula 6,0 y desplaza a un informe corto que trae la respuesta exacta
con 0,85; con la heterogeneidad de tamaños que tiene este corpus, ese sesgo sería
grave. El número de fragmentos solo lo usamos para desempatar, con un peso
pequeño.

**Sin modelos generativos.** No hay reordenamiento por LLM, ni reformulación o
expansión de la consulta, ni filtrado generativo, ni síntesis de fragmentos. Todo
se resuelve sobre vectores, puntuaciones de similitud y metadata.

---

## 8. Salida y evaluación

### Cómo armamos la respuesta

Por consulta devolvemos 3 documentos y 10 fragmentos. El 76,5% de nuestros
fragmentos supera las 250 palabras, así que el límite se activa en la mayoría de
las posiciones.

Se puede partir el fragmento en piezas que respeten el límite, cada una ocupando
su posición, o recortarlo. **Recortamos**, respetando fronteras oracionales. La
decisión salió de medir las dos alternativas sobre el mismo índice: partir baja
el texto entregado de 103.791 a 83.971 palabras por corrida y la cobertura de 10
fragmentos distintos a 6, porque la segunda pieza de un fragmento de 321 palabras
son 71 y se lleva una posición entera con muy poco contenido. Recortar deja las
diez posiciones con 250 palabras cada una.

Dejamos la subdivisión implementada pero desactivada. Si al anotar resulta que la
cola de un fragmento bien posicionado aporta más que un fragmento nuevo peor
posicionado, se activa con un cambio de un valor.

Antes de entregar validamos el archivo: 50 líneas en orden de q001 a q050, tres
documentos y diez fragmentos por consulta, todos los campos presentes y ningún
fragmento por encima del límite de palabras.

### Evaluación interna

El ground truth oficial no es público, así que anotamos uno propio sobre las 50
consultas reales. Eso nos permite ajustar contra la distribución real de la
evaluación en lugar de contra consultas inventadas.

El pool de anotación lo construimos con la unión del top-N de cada encoder por
separado, no con la salida ya fusionada. Si anotáramos solo lo que el sistema
devuelve hoy, cualquier cambio futuro que sacara a la superficie documentos
nuevos aparecería como ruido sin anotar y quedaría penalizado sin merecerlo.

NDCG@10 y F1@3 están implementadas como transcripción literal de las fórmulas del
reto, incluida la normalización del recall por `mín(|D*q|, 3)`, sin la cual una
consulta con cinco documentos relevantes tendría techo 0,6 aunque devolviéramos
los tres mejores posibles. Las dos están cubiertas por pruebas automáticas contra
valores que calculamos a mano, porque son las que van a decidir cada ajuste y una
métrica equivocada no se delata sola: los números seguirían subiendo y bajando de
forma plausible mientras optimizamos en la dirección contraria.

---

## 9. Reproducibilidad

`generador.py` va dentro de la carpeta de entrega, junto al índice que lee y al
archivo de resultados que escribe. Carga los índices ya persistidos, lee el
archivo de consultas y regenera `resultados.jsonl` sin volver a indexar nada.

Es un archivo autónomo: no importa nada del resto del proyecto, porque tiene que
funcionar con solo esa carpeta en manos de quien evalúe. Sus únicas dependencias
son las bibliotecas con las que construimos el índice.

Las semillas están fijadas y las versiones de los modelos ancladas. Además,
tenemos una comprobación automática que corre el entregable y el pipeline de
desarrollo sobre el mismo índice y exige que las salidas sean idénticas, para que
las dos implementaciones no se separen sin que nos demos cuenta.
