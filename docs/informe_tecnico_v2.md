---
title: "Base de Conocimiento Vectorial — Documento Técnico"
subtitle: "CODEFEST AD ASTRA 2026 · Etapa 1 · Entrega v2"
lang: es
---

# Base de conocimiento vectorial — Documento técnico

**CODEFEST AD ASTRA 2026 · Etapa 1 · Entrega v2**

Este documento describe las decisiones de diseño de la base de conocimiento que
entregamos: la estrategia de chunking y su justificación, los encoders y los
criterios con los que los elegimos, el tipo de índice FAISS y el módulo de
recuperación. Casi todas las decisiones salieron de mediciones sobre el corpus
real y las 50 consultas de evaluación, no de argumentos a priori; cuando algo se
decidió por argumento y no llegamos a medirlo, lo decimos.

**Qué cambia respecto a nuestra primera entrega.** Una sola cosa, y es el núcleo
de este documento: las dos listas que pide el reto se construyen ahora con
políticas distintas, porque el reto las evalúa con métricas distintas.

## 1. El corpus, medido

Sobre los 1.826 archivos del inventario oficial, verificados al 100% contra
disco:

| Formato | Archivos | Texto extraído |
|---|---:|---:|
| PDF | 759 | 131,6M caracteres |
| CSV | 26 | 111,9M caracteres |
| PBF | 73 | 7,3M caracteres |
| JSON | 954 | 4,7M caracteres |
| XLSX | 6 | 1,4M caracteres |
| Imagen / TXT | 9 | OCR / texto plano |

Cuatro hechos condicionaron el diseño:

1. **60 PDF sin capa de texto**, 48 de ellos informes escaneados de la Defensoría
   del Pueblo que responden directamente a q033–q050. Exigen OCR.
2. **Asimetría de idioma.** Las 50 consultas están en español; el corpus es
   mayoritariamente inglés, con presencia de portugués. La recuperación
   cross-lingual es el problema central, no un caso borde.
3. **Heterogeneidad extrema de tamaño.** Alertas JSON de 1,3k caracteres conviven
   con atlas de cientos de páginas.
4. **59 nombres de archivo repetidos** en 186 archivos, con contenido distinto
   —verificado por hash—. Esto tiene consecuencias directas sobre el F1@3.

El índice final contiene **64.484 fragmentos** de 1.813 documentos.

## 2. Estrategia de chunking

Estrategia **híbrida**: tamaño acotado por tokens, con frontera obligatoria en
límite oracional y solape del 15%.

**Completitud lingüística (§3.3).** Ninguna oración cruza la frontera entre dos
fragmentos. Eso descarta cortar por número de tokens: los fragmentos se
construyen acumulando oraciones completas y se cierran donde terminó la última
que cabía. La segmentación usa `pysbd` con el idioma detectado por documento —las
abreviaturas que reconoce dependen del idioma, y el 87% del corpus está en
inglés—. El portugués se mapea al segmentador inglés porque `pysbd` no lo
soporta y lo rechaza con excepción; sin ese mapeo, los 7.617 fragmentos en
portugués caían a un camino de respaldo que trataba cada párrafo como una sola
oración.

**Solape del 15%.** Se arrastran las últimas oraciones completas del fragmento
anterior, de modo que la frontera sigue cayendo en límite oracional. Reduce el
riesgo de que la evidencia quede partida entre dos fragmentos sin que ninguno la
contenga entera.

**Presupuesto de tokens.** El chunking se dimensiona contra el más restrictivo de
los dos encoders. La ventana de mE5-large es de 512 tokens e incluye lo que el
fragmento no trae: los tokens especiales y el prefijo `passage: ` que E5 exige.
Un fragmento presupuestado a 512 tokens de contenido entra al modelo con ~518 y
pierde la cola en silencio.

> **Limitación conocida.** El índice que entregamos se construyó con un
> presupuesto de 512 tokens, no de 504: 28.895 de los 64.484 fragmentos (44,8%)
> superan los 504 tokens, con máximo exacto en 512. Para BGE-M3, cuya ventana es
> de 8.192, no tiene ningún efecto. Para mE5-large sí: esos fragmentos entraron
> truncados a su índice. Las métricas que reportamos abajo se obtuvieron con esa
> limitación presente, así que son un piso y no un techo.

**Tope para archivos tabulares.** 30 archivos CSV/XLSX aportaban 91.412 de los
149.571 fragmentos iniciales (61%), y tres exportaciones bibliográficas de PubMed
ocupaban solas el 51,5%. Son listados de referencias biomédicas ajenas a las 50
consultas: consumían más de la mitad del tiempo de codificación y competían en
cada búsqueda contra la evidencia útil. Limitamos a 400 fragmentos por archivo
tabular, donde cada fila es un registro independiente y el valor marginal de la
fila 3.000 es nulo. Los PDF largos y legítimos no se tocan. No excluimos los
archivos: el emparejamiento del F1@3 es por `fuente` y el documento debe seguir
siendo alcanzable.

**Metadata por fragmento (§3.4).** Cada fragmento lleva los ocho campos
obligatorios de la Tabla 1 —`doc_id`, `chunk_id`, `fuente`, `formato`,
`fenomeno`, `posicion`, `num_tokens`, `texto`— más cuatro propios: `idioma`
(decide el segmentador al recortar la salida), `observatorio`, `ruta` y `titulo`.

## 3. Encoders y criterios de selección

| Modelo | Arquitectura | Dim. | Ventana | Licencia | Rol en v2 |
|---|---|---:|---:|---|---|
| `BAAI/bge-m3` | XLM-RoBERTa (encoder) | 1024 | 8192 | MIT | Fragmentos y documentos |
| `intfloat/multilingual-e5-large` | XLM-RoBERTa (encoder) | 1024 | 512 | MIT | Solo documentos |

**Arquitectura (§4.2).** Los dos son encoders. Quedan excluidos por arquitectura
todos los derivados de un backbone autoregresivo —Qwen3-Embedding, e5-mistral—
por más que lideren MTEB, y por licencia Jina v3 (CC-BY-NC-4.0).

**Soporte multilingüe.** Fue el criterio decisivo: consultas en español contra un
corpus mayoritariamente inglés. BGE-M3 se entrena explícitamente para
recuperación cross-lingual (67,8 nDCG@10 en MIRACL sobre 18 idiomas, frente a
65,4 de mE5-large). Los dos operan de forma nativa en español, inglés y
portugués.

**Pooling.** BGE-M3 toma el token CLS; mE5-large promedia con la máscara de
atención. No son intercambiables: usar mean pooling en BGE-M3 por descuido baja
el coseno contra la referencia de 0,999999 a 0,81, un error que parece numérico y
es de configuración. E5 además exige prefijos asimétricos (`query: ` y
`passage: `), aplicados dentro del encoder.

**Qué medimos, y qué descartamos por medición.** Comparamos siete encoders del
mismo catálogo sobre una submuestra con ground truth propio. `me5-base` quedó muy
por debajo (NDCG@10 0,2503, p≈0). `gte-multilingual-base`, elegido por venir de
otra familia de preentrenamiento y aportar más consenso a la fusión, tampoco
superó al par principal. LaBSE entró como control —el argumento era que se
entrenó para alinear frases paralelas y carece de noción de relevancia temática
asimétrica— y el barrido lo confirmó.

## 4. Índice FAISS

`IndexFlatIP` sobre vectores normalizados a norma 1, uno por encoder, en
subcarpetas `encoder_<nombre>/`.

**Por qué exhaustivo y no aproximado.** Con 64.484 vectores de 1024 dimensiones,
el índice ocupa ~252 MB por encoder y una búsqueda exhaustiva sobre las 50
consultas termina en segundos. Un IVF o un HNSW introducirían pérdida de recall a
cambio de una latencia que aquí no es un problema. La Etapa 1 evalúa calidad de
recuperación, no rendimiento, así que cualquier recall sacrificado es puro coste.

**Producto interno como coseno.** Con los vectores normalizados, el producto
interno *es* la similitud coseno (§8.2), sin la raíz cuadrada de la distancia L2.

**Correspondencia con la metadata (§1.4).** La línea *n* de `metadata.jsonl`
describe el vector *n* del índice. Es un invariante del que depende toda la
trazabilidad, y lo verifica el script de empaquetado antes de cerrar la entrega.

## 5. Módulo de recuperación

Aquí está el cambio de esta versión. El reto evalúa las dos listas por separado
—NDCG@10 para fragmentos, F1@3 para documentos (§10.2)— en dos tablas
clasificatorias independientes que después combina por Conteo de Borda (§11).
Nada obliga a que las dos listas salgan del mismo ranking, y medido sobre el
corpus completo no conviene que salgan.

```
                  codificar la consulta con cada encoder
                  buscar en cada índice (pool de 200)
                                  |
        +-------------------------+-------------------------+
        |                                                   |
   FRAGMENTOS: BGE-M3 solo                    DOCUMENTOS: los dos, convexa
   realce por fenómeno                        realce por fenómeno
   deduplicar por texto                       agregación top2 por `fuente`
   diversificar (máx. 3 por doc)               |
   top-10                                     top-3
```

**Por qué separarlas.** Medido sobre el corpus completo, con test de permutación
pareado consulta a consulta:

| Configuración | NDCG@10 | F1@3 |
|---|---:|---:|
| BGE-M3 solo, top2 | **0,7329** | 0,7000 |
| BGE-M3 + mE5-large, convexa, top2 | 0,6597 | **0,7547** |
| BGE-M3 + mE5-large, RRF, top2 | 0,6255 | 0,7413 |
| BGE-M3 + mE5-large, RRF, max | 0,6255 | 0,7213 |
| **v2: fragmentos BGE-M3 / documentos convexa** | **0,7329** | **0,7547** |

Fusionar para los fragmentos cuesta 0,0732 de NDCG@10 (p=0,0012); no fusionar
para los documentos cuesta 0,0547 de F1@3. Con una sola política hay que pagar
uno de los dos precios. La lectura es que mE5-large aporta documentos que BGE-M3
no trae —de ahí la ganancia a nivel documento— pero ensucia el orden fino de los
fragmentos.

**Honestidad estadística.** La ganancia en F1@3 no alcanza significancia
(p=0,1215) y los intervalos de confianza se solapan. La tomamos igual porque su
coste es exactamente cero: la lista de fragmentos no cambia, así que el NDCG@10
es idéntico por construcción, no por medición. Si la ganancia fuera ruido, v2
empata con v1; si es real, gana. No hay escenario en que pierda.

**Fusión convexa.** Normaliza por min-max dentro de cada consulta y cada índice
antes de sumar. Frente a RRF conserva la magnitud —RRF reduce todo a posiciones,
de modo que ganar por 0,30 de coseno y ganar por 0,01 aportan lo mismo—; frente a
sumar cosenos crudos añade la normalización, necesaria porque mE5-large comprime
sus similitudes hacia 0,75–0,92 y BGE-M3 se mueve más abajo.

**Realce por fenómeno.** La correspondencia consulta→fenómeno es conocida, así
que podríamos filtrar en firme. No lo hacemos: q005 y q046 tratan Colombia desde
fenómenos distintos y q027 cruza IA con operaciones espaciales. Aplicamos un
multiplicador suave de 1,05. Medido, su efecto está dentro del ruido.

**Diversificación y deduplicación.** Limitamos a 3 las posiciones que un mismo
documento puede ocupar en el top-10: sin el tope, un único documento erróneo se
lleva la consulta entera. El 3% del corpus (4.445 fragmentos) tiene texto
idéntico —tablas reimpresas, encabezados institucionales—; dos copias gastan dos
posiciones para informar una vez, así que las quitamos de la lista de fragmentos.
No las quitamos de la agregación a documento, porque dos documentos distintos
pueden compartir un texto y descartar uno lo volvería inalcanzable.

**Agregación a documento.** `top2`: la puntuación de un documento es la media de
sus dos mejores fragmentos. Descartamos la suma por su sesgo de longitud —un
documento con cuarenta fragmentos de 0,15 acumula 6,0 y desplaza a un informe
corto con la respuesta exacta a 0,85— y `max` porque, medido, top2 recupera el
F1@3 del fenómeno 2 de 0,7708 a 0,8750: premia que la evidencia esté concentrada
en varios fragmentos buenos sin dejar que muchos flojos ganen por acumulación.

**La clave de agrupación es `fuente`, no `doc_id`.** El emparejamiento del jurado
es por `fuente` (§10.2.1), y el corpus tiene 59 nombres repetidos en 186
archivos: dos `doc_id` con la misma fuente en el top-3 cuentan como un solo
acierto y desperdiciarían una de las tres posiciones. De cada fuente reportamos
el `doc_id` de su mejor fragmento.

**Sin modelos generativos (§4.2, §8.3).** No hay reordenamiento por LLM, ni
reformulación o expansión de consulta, ni filtrado generativo, ni síntesis. Todo
opera sobre vectores, similitudes y metadata. El OCR se hace con Tesseract y no
con un modelo de visión-lenguaje precisamente por esto: aunque el OCR sea
preprocesamiento, el texto que produce termina indexado.

**Medido y descartado.** Probamos realimentación de pseudo-relevancia (Rocchio,
k∈{3,5,10}, β∈{0,3,0,5,0,8}). El mejor resultado sube el NDCG@10 de 0,4879 a
0,5034 con p=0,047, que no sobrevive a la corrección por las catorce
comparaciones de la rejilla; con fusión, además, daña el F1@3 en las ocho
configuraciones. Queda implementado y desactivado. También barrimos la
profundidad del pool: el F1@3 sube de forma monótona hasta 200 candidatos y se
aplana después (400, 800, 1.600 y 3.200 no mejoran nada), así que 200 no es una
cifra heredada sino donde satura.

## 6. Salida y evaluación

**Construcción de la respuesta (§9).** 3 documentos y 10 fragmentos por consulta,
cada fragmento de máximo 250 palabras. El límite es de la *salida*, no del
chunking: se indexa el fragmento completo y el recorte ocurre al escribir. El
corte respeta frontera oracional, con el idioma del fragmento. Si el tope por
documento deja posiciones vacías, una segunda pasada las rellena con fragmentos
aún no emitidos ignorando el tope: un fragmento real de un documento ya
representado informa más que repetir el anterior.

**Evaluación interna.** El ground truth oficial no es público, así que anotamos
uno propio sobre las 50 consultas reales: 1.383 juicios de fragmento sobre 265
documentos, con relevancia graduada 0/1/2. El pool salió de la unión del top-15
de cada encoder por separado, para no sesgar la medida hacia lo que la
configuración actual ya encuentra. Los fragmentos se emparejan por solape de
texto y los documentos por `fuente`, que es como lo hará el jurado.

Resultado de esta entrega sobre ese ground truth, con desglose por fenómeno:

| | NDCG@10 | F1@3 |
|---|---:|---:|
| Global (50 consultas) | **0,7329** | **0,7547** |
| Fenómeno 1 (q001–q016) | 0,6658 | 0,6042 |
| Fenómeno 2 (q017–q032) | 0,8382 | 0,8958 |
| Fenómeno 3 (q033–q050) | 0,6988 | 0,7630 |

El fenómeno 1 es el más flojo en las dos métricas, y no es del sistema sino del
corpus: tiene 6,4 documentos relevantes por consulta contra 9,9 del fenómeno 2, y
15,3 fragmentos con relevancia positiva contra 21,8. Hay menos que encontrar.

## 7. Grafo de conocimiento

No lo construimos. El componente es opcional (§7) y preferimos gastar el tiempo
disponible en medir la política de recuperación contra un ground truth propio,
que es lo que la Etapa 1 evalúa. La entrega no incluye subcarpeta `grafo/`.

## 8. Reproducibilidad

`generador.py` es autónomo: no importa nada del resto del proyecto, porque el
jurado recibe solo este directorio y un script que dependiera de nuestro
repositorio no arrancaría. Carga los índices persistidos, lee el archivo de
consultas y regenera `resultados.jsonl` sin reindexar. Semillas fijadas y
versiones de modelo ancladas.

El precio de esa autonomía es tener la política de recuperación escrita dos
veces —una en el paquete con el que iteramos y otra dentro del entregable—. Para
que no diverjan en silencio, una prueba automática compara ambas
implementaciones sobre un índice sintético en cada ejecución de la batería de
pruebas, y un script de verificación repite la comparación sobre el índice real
como última etapa de la construcción.

```
entrega-v2/
  generador.py          reproduce resultados.jsonl desde esta carpeta
  resultados.jsonl      50 líneas, q001–q050
  informe_tecnico.pdf   este documento
  base_vectorial/
    encoder_bge-m3/     index.faiss + metadata.jsonl
    encoder_me5-large/  index.faiss + metadata.jsonl
```

Los dos índices son necesarios: el de BGE-M3 decide los fragmentos y ambos
deciden los documentos. Con un solo índice disponible, `generador.py` no falla:
las dos listas caen sobre el que haya y el sistema se comporta como nuestra
primera entrega.
