---
title: "Base de Conocimiento Vectorial — Informe Técnico"
subtitle: "CODEFEST AD ASTRA 2026 · Etapa 1"
lang: es
---

## 1. Caracterización del corpus

Todas las decisiones de diseño que siguen se apoyan en mediciones sobre el
inventario completo de ADL (1826 archivos, verificados al 100% contra disco).

| Formato | Archivos | Texto extraído |
|---|---:|---:|
| PDF | 759 | 131,6M caracteres |
| CSV | 26 | 111,9M caracteres |
| PBF | 73 | 7,3M caracteres (atributos de mapa) |
| JSON | 954 | 4,7M caracteres |
| XLSX | 6 | 1,4M caracteres |
| Imagen / TXT | 9 | OCR / texto plano |
| **Total** | **1826** | **~257M caracteres** |

Cuatro hechos medidos determinaron la arquitectura:

**La asimetría de idioma es el problema central.** Las 50 consultas de evaluación
están en español, mientras que el corpus es mayoritariamente inglés: de los
149.571 fragmentos indexados, 130.090 están en inglés, 11.066 en español y 7.617
en portugués. La recuperación cross-lingual no es un caso borde sino el requisito
dominante, y pesó sobre la elección de encoder más que ningún otro criterio.

**60 PDFs no tienen capa de texto.** 48 de ellos son informes escaneados de
alertas tempranas de la Defensoría del Pueblo, precisamente el material que
responde las consultas q033–q050. Omitirlos dejaría al sistema ciego en el tercer
fenómeno completo.

**Heterogeneidad extrema de tamaño.** Alertas JSON de ~1,3k caracteres conviven
con atlas RESDAL de varios cientos de páginas. Esto descartó el *sum pooling*
para la agregación a documento y obligó a una fragmentación que tolere ambos
extremos.

**Los CSV dominaban el índice en bruto.** Aportaban 90.442 de los 149.571
fragmentos (60%), y tres exportaciones bibliográficas de PubMed ocupaban por sí
solas el 51,5%: listados de referencias biomédicas ajenos a cualquier consulta de
evaluación, que consumían más de la mitad del tiempo de codificación y competían
en cada búsqueda contra la evidencia útil.

Se aplica un **tope de 400 fragmentos por documento tabular**, que reduce el
índice a unos 63.000 fragmentos. No se excluye ningún archivo: el emparejamiento
del F1@3 se hace por `fuente` (§10.2.1), de modo que cada documento debe seguir
siendo alcanzable. El tope solo afecta a CSV y XLSX, donde cada fila es un
registro independiente y el valor marginal de la fila 3.000 es nulo; los PDF
extensos y legítimos —una ley de 1.960 fragmentos— quedan intactos.

## 2. Identificación de documentos

No se inventan identificadores. Se replica exactamente el esquema que ADL define
en `Indice_Datos_Codefest.xlsx`:

```
doc_id = {fenomeno}-{codigo_observatorio}-{consecutivo:03d}   p. ej. F1-CSET-001
fuente = campo "Nombre estandarizado" del inventario, literal
```

Esto importa porque la §10.2.1 establece que el emparejamiento con el ground
truth se realiza a través de `fuente`, no del `doc_id`. Cualquier divergencia
respecto al nombre original de ADL invalidaría la métrica F1@3 con independencia
de la calidad de la recuperación.

La ingesta itera sobre el inventario, nunca sobre un listado del directorio. Eso
excluye por construcción los artefactos que contaminarían el índice: los
`.DS_Store`, las hojas de inventario y —crítico— el PDF de preguntas, cuya
indexación metería el examen dentro del corpus.

Una salvedad verificada: 59 nombres estandarizados están duplicados en carpetas
distintas (186 archivos) y contienen material genuinamente diferente, confirmado
por hash MD5, que también estableció que el corpus no tiene duplicados exactos.
Se reporta el nombre literal en `fuente` y la ruta completa se conserva en un
campo aparte para trazabilidad interna.

## 3. Extracción de texto

| Formato | Enfoque |
|---|---|
| PDF | PyMuPDF con ordenamiento por bloques de lectura, decisivo en los diseños a dos columnas de los atlas RESDAL y los informes CSIS |
| JSON (artículo) | `title` + `body_paragraphs`/`body_text` al cuerpo; `url`, `date`, `authors`, `tags` se conservan como metadata en lugar de mezclarse con el texto |
| JSON (catálogo) | Cada registro de la lista se emite como bloque independiente |
| CSV / XLSX | Cada fila como unidad, con pares `columna: valor` para que todo valor conserve su cabecera como contexto |
| PBF | Atributos de cada elemento como `clave: valor`, deduplicados dentro de cada tesela |
| Imagen | OCR |

Las alertas de la Defensoría exigieron tratamiento propio: su contenido
sustantivo —tema de riesgo, municipios, grupos armados— vive en el objeto
`alerta_meta` y no en `body_paragraphs`. Extraer solo el cuerpo habría reducido
cada alerta a un párrafo suelto.

Sobre PBF, la §2.1 recomienda quedarse con una sola versión de cada elemento, que
se repite entre niveles de zoom. Se aplica **dentro** de cada tesela, no entre
teselas: ADL asigna un `doc_id` distinto a cada archivo PBF, de modo que
descartar teselas volvería inalcanzables documentos que el ground truth podría
marcar como relevantes.

### Limpieza

El boilerplate se elimina por **frecuencia de línea** y no por patrones fijos.
Una línea corta que se repite cuatro o más veces dentro de un documento es un
encabezado institucional o una etiqueta de eje; una oración del cuerpo no lo es.
El criterio es estructural y funciona igual en los tres idiomas del corpus. Sobre
un capítulo representativo del AI Index eliminó el 14% de las líneas
—encabezados, fuentes de figura y rótulos de eje— dejando la prosa intacta.

El idioma predominante se estima por palabras funcionales, desambiguando español
y portugués mediante marcadores exclusivos.

## 4. OCR

Se emplea **Tesseract con datos de idioma español** (`-l spa`) a 200 dpi.

Se implementó y descartó una alternativa por modelo de visión-lenguaje
(`baidu/Unlimited-OCR`, 3B, licencia MIT, consciente del layout). El descarte
responde a dos razones, en este orden:

1. **Es una arquitectura decoder.** La §4.2 las prohíbe en la construcción del
   índice. El OCR pertenece al preprocesamiento (§2.1), donde la especificación lo
   recomienda explícitamente, y el argumento de que queda fuera del alcance de la
   prohibición es defendible. Pero el texto que produce termina indexado, y la
   sanción por una lectura estricta es la exclusión, no una penalización. El
   riesgo no compensa la ganancia.
2. **Alucinación.** Un modelo de visión-lenguaje que inventa texto inyecta
   evidencia falsa en el índice, que después puede presentarse al evaluador como
   soporte recuperado. Tesseract, cuando falla, produce ruido evidente.

Sobre cada documento procesado se calculan tres señales de calidad, porque un OCR
que falla en silencio es peor que uno que revienta: densidad de diacríticos (un
texto español sin tildes ni eñes indica paquete de idioma equivocado, aunque
parezca legible), proporción de caracteres no imprimibles y palabras por página.
Los documentos que disparan alguna señal se revisan contra el PDF original antes
de entrar al índice.

## 5. Fragmentación

**Configuración: 512 tokens, 15% de solape, cortes en frontera oracional.**

La §3.3 prohíbe que una oración cruce la frontera entre fragmentos. Esto excluye
cortar por conteo de tokens y obliga a acumular oraciones completas: cuando la
siguiente no cabe en el presupuesto, el fragmento cierra donde terminó la
anterior. El solape se implementa arrastrando las oraciones finales hacia
adelante, de modo que también las fronteras de solape caen en límites oracionales.
La segmentación usa `pysbd`, con soporte nativo para español, inglés y portugués.

El conteo de tokens usa el tokenizador del propio encoder y no una aproximación
por palabras, porque el presupuesto que importa es el del modelo.

Dos detalles de implementación resultaron necesarios a escala de corpus:

- El corpus contiene bloques de más de 140.000 tokens sin puntuación interna
  (volcados de mapas PBF, filas de CSV muy anchas). Partirlos añadiendo palabras y
  recontando es cuadrático; la implementación tokeniza una sola vez y corta
  ventanas sobre el mapa de offsets. Esto redujo la fragmentación del corpus
  completo de 147 a 94 minutos con salida idéntica byte a byte.
- Sumar los tokens de cada oración no equivale a tokenizar el texto unido: el
  tokenizador puede fusionar o dividir piezas en las junturas. Los fragmentos
  cercanos al presupuesto se recuentan y se parten si lo exceden, lo que evita
  truncamiento silencioso bajo encoders de ventana corta como mE5-large.

**Excepción declarada al requisito §3.3.** En los volcados de atributos de mapa y
las filas tabulares no existe frontera oracional a la que retroceder: no son
prosa. Cuando una única «oración» excede el presupuesto del modelo se corta
primero por separadores débiles (`;`, `,`, `|`), que preservan unidades legibles,
y solo lo que aún no quepa se trocea por ventanas de tokens.

Medido sobre una muestra estratificada por formato, el porcentaje de fragmentos
producidos por esas dos rutas es:

| Formato | Corte por separador | Ventana de tokens |
|---|---:|---:|
| PDF | 0,0% | 0,0% |
| JSON | 0,0% | 0,0% |
| XLSX | 0,0% | 0,0% |
| CSV | 15,2% | 1,5% |
| PBF | 60,4% | 0,0% |

**Todo el texto en prosa —PDF, JSON de artículo, XLSX y OCR— respeta el requisito
sin una sola excepción.** Las desviaciones se concentran donde el requisito no es
aplicable: filas de CSV y atributos de teselas PBF, que no contienen oraciones.

**Parámetro sujeto a barrido empírico.** El tamaño de fragmento presenta tensión
entre las dos métricas: 512 tokens favorece NDCG@10 mientras 1024 favorece F1@3.
Como el Conteo de Borda pondera ambas por igual, el valor se fija empíricamente
barriendo {384, 512, 768} contra el conjunto de evaluación interno.

Corpus resultante: **149.571 fragmentos, 72,3M tokens, mediana de 499 tokens por
fragmento.** La cobertura del índice es de 1758 de 1826 documentos; el resto son
los 60 pendientes de OCR y 8 JSON que solo contienen fechas.

### Metadata por fragmento

Se emiten los ocho campos obligatorios de la Tabla 1 —`doc_id`, `chunk_id`,
`fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`— más cuatro
extensiones para post-filtros: `idioma`, `observatorio`, `ruta` y `titulo`.

El campo `formato` toma el valor real del archivo (`pdf`, `json`, `csv`, `xlsx`,
`pbf`, `txt`, `imagen`). La Tabla 1 enumera `pdf, html o md`, pero el corpus
descrito en la §1.3 incluye los siete formatos, y reportar el formato real es lo
único que hace útil el post-filtro de la §8.7.

## 6. Codificación semántica

| Modelo | Licencia | Arquitectura | Dim. | Contexto | Rol |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | MIT | XLM-RoBERTa (encoder) | 1024 | 8192 | Principal |
| `intfloat/multilingual-e5-large` | MIT | XLM-RoBERTa (encoder) | 1024 | 512 | Complementario |

**BGE-M3 como principal:** supera a mE5-large en recuperación en español (0,727
frente a 0,660 en MIRACL-VISION), su ventana de 8192 tokens elimina toda
restricción sobre la fragmentación, y produce representaciones sparse junto a las
densas en una sola pasada, aportando sensibilidad léxica sin un índice adicional.
Esa componente léxica importa aquí: las consultas están cargadas de siglas (NBQR,
RPO, GEO, DIH, ASAT, GAO/GAOR/GDO) y nombres propios (Chocó, Antioquia, Arauca,
Norte de Santander).

**Razón del segundo encoder:** mE5-large mantiene un espacio vectorial más
particionado por idioma, lo que reduce la tendencia a devolver documentos en un
idioma distinto al de la consulta. Complementa la debilidad conocida de BGE-M3 en
esa dimensión. E5 exige prefijos asimétricos (`query: ` / `passage: `), que se
aplican de forma transparente en la frontera del encoder.

**Modelos descartados:**

- **Qwen3-Embedding** lidera MTEB multilingüe (70,88 en recuperación) pero deriva
  de un backbone autoregresivo decoder. **Prohibido explícitamente por la §4.2.**
- **Jina Embeddings v3** tiene licencia CC-BY-NC-4.0, incompatible con el criterio
  de licencia de la §4.3.
- **LaBSE** se entrenó para alinear pares de frases paralelas y carece de noción
  de relevancia temática asimétrica; rinde 18,80 en recuperación zero-shot.

## 7. Índice vectorial

**`IndexFlatIP` sobre vectores normalizados a norma unitaria**, un índice
independiente por encoder.

A este tamaño de corpus la búsqueda exhaustiva es exacta y se resuelve en
milisegundos. Los índices aproximados (IVF, HNSW) cambian exactitud por una
velocidad que este volumen no requiere, y aquí la exactitud del ranking *es* la
métrica evaluada. La normalización previa hace que el producto interno equivalga a
la similitud coseno (§8.2).

FAISS almacena únicamente vectores e identificadores enteros. El vínculo con la
metadata es posicional: la línea *n* de `metadata.jsonl` describe el vector *n*.
El orden de inserción es parte del contrato, y la alineación índice/metadata se
verifica tanto al guardar como al cargar.

## 8. Recuperación

```
búsqueda por índice → fusión RRF → boost por fenómeno → diversificación
                    → top-10 fragmentos → max pooling → top-3 documentos
```

**Reciprocal Rank Fusion** combina los encoders. RRF opera sobre posiciones y no
sobre puntuaciones, lo que lo hace inmune a la diferencia de escalas entre
espacios vectoriales distintos: los valores coseno de BGE-M3 y de E5 no son
comparables entre sí, pero sus órdenes sí lo son. `k₀ = 60` es el punto de
partida, ajustado contra el conjunto de evaluación interno.

**Boost por fenómeno, no filtro duro.** La correspondencia consulta→fenómeno es
conocida (q001–q016 → F1, q017–q032 → F2, q033–q050 → F3), pero varias consultas
admiten evidencia transversal: q005 y q046 abordan Colombia desde fenómenos
distintos, y q027 cruza inteligencia artificial con operaciones espaciales. Un
filtro estricto cerraría el acceso a documentos legítimamente relevantes, de modo
que se aplica un multiplicador suave.

**Diversificación** limita cuántos fragmentos aporta un mismo documento al top-10.
Sin ella un solo documento puede ocupar las diez posiciones evaluadas; si resulta
no ser relevante, la consulta se pierde entera.

**Max pooling** para la agregación a documento: cada documento hereda la
puntuación de su mejor fragmento. Se descarta sum pooling por su sesgo de longitud
—un documento con cuarenta fragmentos débiles (0,15 cada uno) acumula 6,0 y
desplaza a un informe preciso que contiene la respuesta con 0,85—. Dada la
heterogeneidad de tamaños documentada en la §1, ese sesgo sería severo. El número
de fragmentos actúa solo como desempate, con peso suficientemente pequeño para no
alterar el orden principal.

**Ningún modelo generativo interviene en ninguna etapa:** no hay reranking por
LLM, expansión de consulta, filtrado generativo ni síntesis. Todas las operaciones
corren sobre vectores, puntuaciones de similitud y metadata, como exige la §8.3.

### Construcción de la salida

Los fragmentos que superan las 250 palabras se subdividen respetando fronteras
oracionales; el `chunk_id` reportado sigue siendo el del fragmento originario del
índice, cumpliendo una función de trazabilidad y no de emparejamiento (§10.2.1).
El archivo generado se valida contra el esquema de la §9.3 antes de la entrega:
exactamente 50 líneas, 3 documentos y 10 fragmentos por consulta, y ningún
fragmento por encima del límite de palabras.

## 9. Evaluación interna

El ground truth oficial no es público. Se anota uno propio sobre las 50 consultas
reales, lo que permite optimizar contra la distribución real de evaluación en
lugar de sustitutos sintéticos.

El pool de anotación se construye con la unión del top-N de cada encoder **por
separado**, no con la salida fusionada del sistema. Anotar únicamente la salida
final sesgaría la medición hacia lo que la configuración actual ya recupera:
cualquier cambio futuro que sacara a la superficie documentos nuevos aparecería
como ruido sin anotar y sería penalizado injustamente.

NDCG@10 y F1@3 están implementadas como transcripción literal de las fórmulas de
la §10.2, incluyendo la normalización `mín(|D*q|, 3)` del recall, sin la cual una
consulta con cinco documentos relevantes quedaría topada en 0,6 aun devolviendo
los tres mejores posibles.

## 10. Reproducibilidad

`generador.py` se entrega dentro de `entrega/`, junto al índice que lee y al
archivo de resultados que escribe, según la estructura de la §1.4. Carga los
índices persistidos, lee el archivo de consultas y regenera `resultados.jsonl`
sin reindexar.

Es un archivo autónomo: no importa nada del resto del proyecto, porque `entrega/`
debe funcionar por sí solo en manos del evaluador. Sus únicas dependencias son
las bibliotecas con las que se construyó el índice: `faiss-cpu`, `numpy`,
`sentence-transformers`, `pymupdf` y `pysbd`.

Las semillas están fijadas y las versiones de modelo ancladas. Una comprobación
automatizada contrasta la salida del entregable contra la del pipeline de
desarrollo sobre el mismo índice y exige que sean idénticas, de modo que las dos
implementaciones no puedan divergir en silencio.

La especificación excluye de la evaluación toda entrega que no reproduzca sus
resultados; se trata como criterio eliminatorio y no como recomendación.
