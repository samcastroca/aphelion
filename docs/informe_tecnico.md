---
title: "Base de Conocimiento Vectorial — Documento Técnico"
subtitle: "CODEFEST AD ASTRA 2026 · Etapa 1"
lang: es
---

La §1.4 pide que este documento describa cuatro decisiones: la **estrategia de
chunking y su justificación**, los **encoders seleccionados y los criterios de
elección**, el **tipo de índice FAISS empleado** y, de aplicar, la **descripción
del grafo de conocimiento**. Las secciones 2 a 5 responden a esas cuatro en ese
orden. La sección 1 expone las mediciones sobre el corpus de las que dependen
todas ellas, y las secciones 6 a 9 documentan el preprocesamiento, el módulo de
recuperación, el formato de salida y la reproducibilidad.

## 1. El corpus, medido

Todas las decisiones que siguen se apoyan en mediciones sobre el inventario
completo que provee ADL (1826 archivos, verificados al 100% contra disco), no en
estimaciones previas.

| Formato | Archivos | Texto extraído |
|---|---:|---:|
| PDF | 759 | 131,6M caracteres |
| CSV | 26 | 111,9M caracteres |
| PBF | 73 | 7,3M caracteres |
| JSON | 954 | 4,7M caracteres |
| XLSX | 4 | 1,4M caracteres |
| Imagen / TXT | 9 | OCR / texto plano |
| **Total** | **1826** | **~257M caracteres** |

Cuatro hechos determinaron la arquitectura.

**La recuperación cross-lingual es el problema central, no un caso borde.** Las
50 consultas de evaluación están en español y el corpus es mayoritariamente
inglés: de los fragmentos indexados, 130.090 están en inglés, 11.066 en español y
7.617 en portugués. El criterio de **soporte multilingüe** de la §4.3 pesó sobre
la elección de encoder más que ningún otro.

**60 PDFs carecen de capa de texto.** 48 son informes escaneados de alertas
tempranas de la Defensoría del Pueblo, el material que responde las consultas
q033–q050. Omitirlos dejaría al sistema ciego en el fenómeno 3 completo.

**La heterogeneidad de tamaño es extrema.** Alertas JSON de ~1,3k caracteres
conviven con atlas RESDAL de varios cientos de páginas. Esto descartó el sum
pooling en la agregación a nivel de documento (§8.6) y obligó a una estrategia de
chunking que tolere ambos extremos.

**Los archivos tabulares dominaban el índice.** 30 archivos CSV y XLSX aportaban
90.442 de 149.571 fragmentos, y tres exportaciones bibliográficas de PubMed
ocupaban por sí solas el 51,5%. Son listados de referencias biomédicas ajenos a
las 50 consultas: consumían más de la mitad del coste de codificación y competían
en cada búsqueda contra la evidencia útil.

---

## 2. Estrategia de chunking y su justificación

**Configuración: 512 tokens, 15% de superposición, cortes en frontera oracional.**

### La estrategia, en los términos de la §3.2

Es una **estrategia híbrida**, de las que la §3.2 admite explícitamente, que
combina tres de las descritas:

- **Por tamaño fijo de tokens** fija el presupuesto en 512, que es lo que acota el
  coste de codificación y el tamaño del índice.
- **Por oración** define dónde puede caer el corte. El presupuesto nunca parte una
  oración: cuando la siguiente no cabe, el fragmento cierra donde terminó la
  anterior.
- **Semántica con superposición** aporta el solape del 15%, implementado
  arrastrando hacia adelante las oraciones finales del fragmento previo, de modo
  que también las fronteras de solape caen en límites oracionales.

Se descarta la **jerárquica o estructural**: el corpus llega en siete formatos y
solo los PDF conservan señales estructurales fiables; aplicarla produciría
criterios distintos por formato sin una ganancia que lo justifique.

### Cumplimiento del requisito de completitud lingüística (§3.3)

El requisito prohíbe que una oración se extienda a través de la frontera entre
dos fragmentos. Esto excluye cortar por conteo de tokens y obliga a acumular
oraciones completas. La segmentación usa `pysbd`, con soporte nativo para
español, inglés y portugués. El conteo de tokens usa el tokenizador del propio
encoder y no una aproximación por palabras, porque el presupuesto que importa es
el del modelo.

Medido sobre una muestra estratificada por formato, el porcentaje de fragmentos
producidos por rutas que no cortan en frontera oracional es:

| Formato | Corte por separador débil | Ventana de tokens |
|---|---:|---:|
| PDF | 0,0% | 0,0% |
| JSON | 0,0% | 0,0% |
| XLSX | 0,0% | 0,0% |
| CSV | 15,2% | 1,5% |
| PBF | 60,4% | 0,0% |

**Todo el texto en prosa —PDF, JSON de artículo, XLSX y OCR— cumple el requisito
sin una sola excepción.** Las desviaciones se concentran donde el requisito no es
aplicable: filas de CSV y volcados de atributos de teselas PBF, que no contienen
oraciones. Cuando una única «oración» de ese tipo excede el presupuesto del
modelo se corta primero por separadores débiles (`;`, `,`, `|`), que preservan
unidades legibles, y solo lo que aún no quepa se trocea por ventana de tokens.

### Justificación del tamaño

Evaluaciones comparativas sobre corpus técnicos sitúan el fragmento de 512 tokens
con solape alineado a oraciones como el punto de partida defendible: un barrido
de 2026 sobre siete estrategias lo situó primero, y un estudio previo sitúa el
pico de fidelidad en 1024, de modo que el rango 512–1024 es el razonable.

Se descarta el chunking semántico puro: pese a su buen desempeño a nivel de
fragmento, colapsa a nivel de documento al producir fragmentos de ~43 tokens que
fragmentan el contexto — precisamente la métrica F1@3 que evalúa la §10.2.2.

El tamaño presenta tensión entre las dos métricas: 512 favorece NDCG@10 mientras
1024 favorece F1@3. Como el Conteo de Borda (§11.2) pondera ambas por igual, el
valor se fija empíricamente barriendo {384, 512, 768} contra el conjunto de
evaluación interno. La superposición entra al mismo barrido: evidencia reciente
cuestiona que aporte beneficio medible, y cuesta un 15% adicional de vectores.

### Tope sobre los formatos tabulares

Se aplica un tope de **400 fragmentos por documento tabular**, que reduce el
índice de 149.571 a unos 63.000 fragmentos. Es un **post-filtro sobre el campo
`formato`** de la metadata, en el sentido de la §8.7, aplicado en la indexación.

No se excluye ningún archivo: la §10.2.1 establece que el emparejamiento a nivel
de documento se realiza a través del campo `fuente`, de modo que cada documento
debe seguir siendo alcanzable. El tope solo afecta a CSV y XLSX, donde cada fila
es una unidad de fragmentación independiente (§2.1) y el valor marginal de la
fila 3.000 es nulo. Los PDF extensos y legítimos —una ley de 1.960 fragmentos—
quedan intactos.

### Metadata obligatoria por fragmento (§3.4)

Se emiten los ocho campos de la Tabla 1 —`doc_id`, `chunk_id`, `fuente`,
`formato`, `fenomeno`, `posicion`, `num_tokens`, `texto`— más cuatro extensiones
de las que la §3.4 permite: `idioma`, `observatorio`, `ruta` y `titulo`.

El campo `formato` toma el valor real del archivo (`pdf`, `json`, `csv`, `xlsx`,
`pbf`, `txt`, `imagen`). La Tabla 1 enumera `pdf, html o md`, pero la §1.3 lista
los siete formatos que ADL entrega, y reportar el formato real es lo único que
hace utilizable el post-filtro por metadata de la §8.7.

**Corpus resultante: ~63.000 fragmentos, mediana de 499 tokens por fragmento**,
sobre 1758 de los 1826 documentos. Los 68 restantes son los 60 que requieren OCR
y 8 JSON que solo contienen fechas.

---

## 3. Encoders seleccionados y criterios de elección

Se emplean **dos encoders**, opción que la §4.4 admite explícitamente, cada uno
con su propio índice FAISS independiente.

| Modelo | Arquitectura | Dim. | Contexto | Licencia | Rol |
|---|---|---:|---:|---|---|
| `BAAI/bge-m3` | XLM-RoBERTa (encoder) | 1024 | 8192 | MIT | Principal |
| `intfloat/multilingual-e5-large` | XLM-RoBERTa (encoder) | 1024 | 512 | MIT | Complementario |

### Justificación contra los seis criterios de la §4.3

**Soporte multilingüe.** Es el criterio dominante, porque las consultas están en
español y el 87% de los fragmentos en inglés. BGE-M3 se entrena explícitamente
para recuperación cross-lingual mediante autodestilación y rinde 67,8 nDCG@10 en
MIRACL sobre 18 idiomas, frente a 65,4 de mE5-large. Ambos operan de forma nativa
en los tres idiomas del corpus.

**Dimensionalidad del vector.** Ambos producen 1024 dimensiones. Con ~63.000
fragmentos el índice ocupa unos 250 MB por encoder, holgado para búsqueda
exhaustiva. La §4.3 advierte que dimensiones más altas no garantizan mejor
rendimiento, y es la razón por la que no se buscó un modelo de mayor dimensión.

**Longitud máxima de entrada.** La ventana de 8192 tokens de BGE-M3 elimina toda
restricción sobre la estrategia de chunking. mE5-large se limita a 512, que es
justamente el presupuesto elegido: el chunking se diseñó para no exceder el
límite del más restrictivo de los dos, de modo que ningún fragmento sufre
truncamiento silencioso en ninguno de los dos índices.

**Rendimiento en benchmarks.** Ambos son modelos de recuperación densa, no
optimizados para clasificación o similitud de pares, que es lo que la §4.3
prefiere. Las cifras de MIRACL citadas arriba son de recuperación.

**Licencia.** Ambos son MIT, una de las que la §4.3 prefiere explícitamente.

**Eficiencia computacional.** 568M y 560M parámetros respectivamente; el índice
se construye una sola vez y la codificación de una consulta es de milisegundos.
En hardware sin GPU el corpus completo tardaría días, por lo que la indexación se
resuelve sobre GPU y el pipeline es agnóstico del dispositivo.

### Papel del segundo encoder

mE5-large mantiene un espacio vectorial más particionado por idioma, lo que
reduce la tendencia a devolver documentos en un idioma distinto al de la
consulta. Complementa la debilidad conocida de BGE-M3 en esa dimensión, que es el
tercer supuesto de la §4.4: mejorar la robustez combinando rankings de espacios
vectoriales diferentes. E5 exige prefijos asimétricos (`query: ` / `passage: `),
aplicados de forma transparente en la frontera del encoder.

### Modelos descartados

- **Qwen3-Embedding**, líder en MTEB multilingüe, deriva de un backbone
  autoregresivo decoder. **Prohibido explícitamente por la §4.2.**
- **Jina Embeddings v3** tiene licencia CC-BY-NC-4.0, incompatible con el criterio
  de licencia de la §4.3.
- **LaBSE** se entrenó para alinear pares de frases paralelas y carece de noción
  de relevancia temática asimétrica; rinde 18,80 en recuperación zero-shot.

### Sobre las cabezas sparse y multi-vector de BGE-M3

El modelo produce representaciones densas, sparse y multi-vector en la misma
pasada, y combinarlas eleva su nDCG@10 en MIRACL de 67,8 a 70,0. Ninguna de las
tres es un decoder, de modo que usarlas no rozaría la §4.2. **Se usa solo la
cabeza densa.** La recuperación léxica rinde peor precisamente en escenarios
cross-lingual, donde el solapamiento de vocabulario entre consulta y documento es
mínimo, que es la situación dominante de este corpus. Cabe esperar que ayude en
las siglas y topónimos que sí cruzan idiomas (NBQR, ASAT, Chocó, Arauca) y
estorbe en el resto; sin ground truth no se puede saber cuál efecto domina, y
adoptar una componente de recuperación sin medir su efecto contradice el criterio
que gobierna el resto de estas decisiones.

---

## 4. Tipo de índice FAISS empleado

**`IndexFlatIP` sobre vectores normalizados a norma unitaria**, un índice
independiente por encoder, según exige la §4.4.

De los tres tipos que describe la §5.2, se elige el plano por la razón que la
propia especificación señala: para el volumen de documentos esperado en este
reto, un índice plano es suficiente y garantiza resultados exactos.
`IndexIVFFlat` e `IndexHNSW` intercambian exactitud por una velocidad que
~63.000 vectores no requieren, y aquí la exactitud del ranking **es** la métrica
evaluada. Con este volumen la búsqueda exhaustiva se resuelve en milisegundos.

Se elige el producto interno (`IP`) y no L2 porque, con los vectores normalizados
previamente, el producto interno equivale a la similitud coseno (§8.2), que es la
medida que la §8.2 establece para comparar la consulta con el índice.

**Relación con el almacén de metadata (§5.3).** FAISS almacena únicamente los
vectores y sus identificadores enteros internos. El vínculo con la metadata es
posicional: la línea *n* de `metadata.jsonl` describe el vector *n* del índice,
como exige la §1.4. El orden de inserción es parte del contrato, y la alineación
entre índice y metadata se verifica tanto al persistir como al cargar.

**Persistencia (§5.4).** El índice se serializa con `faiss.write_index()` y es
directamente cargable con `faiss.read_index()` sin dependencias adicionales.

---

## 5. Grafo de conocimiento

**No se implementa.** El componente de la §7 es opcional y este equipo prioriza
la calidad de la base vectorial, que es lo que la §1.2 establece que la Etapa 1
evalúa. En consecuencia, la entrega no incluye la subcarpeta `grafo/` ni el
archivo `grafo.graphml`, y la §8.5 no interviene en el módulo de recuperación.

---

## 6. Preprocesamiento de las fuentes

### Extracción de texto (§2.1)

| Formato | Enfoque |
|---|---|
| PDF | PyMuPDF preservando el orden de lectura de los párrafos, decisivo en los diseños a dos columnas de los atlas RESDAL y los informes CSIS |
| JSON (artículo) | `title` + `body_paragraphs`/`body_text` concatenados en orden; `url`, `date`, `authors`, `tags` se conservan como metadata del documento en lugar de mezclarse con el cuerpo |
| JSON (catálogo) | Cada registro de la lista se emite como bloque independiente |
| CSV / XLSX | Cabecera primero y luego cada fila como unidad, en pares `columna: valor`, de modo que todo valor conserva el nombre de su columna como contexto |
| PBF | Se recorren las capas y, dentro de cada una, los elementos del mapa, volcando sus atributos como pares `atributo: valor` |
| Imagen | OCR |

Sobre PBF, la §2.1 recomienda quedarse con una sola versión de cada elemento
para no duplicar la data. Se aplica **dentro** de cada tesela, no entre teselas:
ADL asigna un `doc_id` distinto a cada archivo PBF, de modo que descartar teselas
volvería inalcanzables documentos que el ground truth podría marcar relevantes.

Las alertas de la Defensoría exigieron tratamiento propio: su contenido
sustantivo —tema de riesgo, municipios, grupos armados— vive en el objeto
`alerta_meta` y no en `body_paragraphs`. Extraer solo el cuerpo habría reducido
cada alerta a un párrafo suelto.

### OCR

Se emplea **Tesseract** con los paquetes de español e inglés a 200 dpi. No todos
los escaneados están en español: 48 son de la Defensoría, pero el resto son
informes de CSET y CSIS.

Se implementó y descartó una alternativa por modelo de visión-lenguaje
(`baidu/Unlimited-OCR`, licencia MIT, consciente del layout). El OCR pertenece al
preprocesamiento, donde la §2.1 lo recomienda explícitamente, y la prohibición de
la §4.2 gobierna la generación de embeddings y el módulo de recuperación. Aun
así, el texto que produce termina indexado, y la sanción por una lectura estricta
de la §4.2 sería la exclusión. A eso se suma que un modelo generativo que alucina
inyecta evidencia falsa en el índice, que podría acabar presentada al evaluador
como soporte recuperado; Tesseract, cuando falla, produce ruido evidente.

Sobre cada documento se miden tres señales de calidad, porque un OCR que falla en
silencio es peor que uno que revienta: densidad de diacríticos —solo cuando el
texto resulta estar en español—, proporción de caracteres no imprimibles y
palabras por página. Lo que dispare alguna señal se revisa contra el original
antes de entrar al índice.

### Limpieza y normalización (§2.2)

Se normaliza la codificación a UTF-8, se eliminan caracteres de control y
espacios redundantes, y se detecta el idioma predominante, que se almacena como
metadata para post-filtros.

Los elementos repetitivos sin valor informativo se eliminan por **frecuencia de
línea** y no por patrones fijos. Una línea corta que se repite cuatro o más veces
dentro de un documento es un encabezado, un pie o un rótulo de eje; una oración
del cuerpo no lo es. El criterio es estructural y funciona igual en los tres
idiomas del corpus. Sobre un capítulo representativo del AI Index eliminó el 14%
de las líneas dejando la prosa intacta.

### Identificación de documentos (§2.3)

No se inventan identificadores. Se replica el esquema que ADL define en
`Indice_Datos_Codefest.xlsx`:

```
doc_id = {fenomeno}-{codigo_observatorio}-{consecutivo:03d}   p. ej. F1-CSET-001
fuente = campo "Nombre estandarizado" del inventario, literal
```

Importa porque la §10.2.1 establece que el emparejamiento a nivel de documento se
realiza a través de `fuente`, no del `doc_id` que cada equipo asigna. Cualquier
divergencia respecto al nombre original invalidaría el F1@3 con independencia de
la calidad de la recuperación.

La ingesta itera sobre el inventario y nunca sobre un listado del directorio. Eso
excluye por construcción lo que contaminaría el índice: los `.DS_Store`, las
hojas de inventario y —crítico— el PDF de preguntas, cuya indexación metería el
conjunto de evaluación dentro del corpus.

Una salvedad verificada: 59 nombres estandarizados están duplicados en carpetas
distintas (186 archivos) con contenido genuinamente diferente, confirmado por
hash MD5, que también estableció que el corpus no contiene duplicados exactos. Se
reporta el nombre literal en `fuente` y la ruta completa se conserva aparte para
trazabilidad interna.

---

## 7. Módulo de recuperación

```
búsqueda por índice → fusión RRF → post-filtro por fenómeno → diversificación
                    → top-10 fragmentos → max pooling → top-3 documentos
```

La consulta se codifica con **el mismo encoder** empleado en la indexación
(§8.1), con el prefijo de instrucción que cada modelo requiere, y se normaliza
igual que los vectores del índice.

**Combinación de múltiples bases vectoriales (§8.4).** Se emplea **Reciprocal
Rank Fusion** con `k₀ = 60`, el valor típico que la §8.4 indica. RRF combina
posiciones en lugar de puntuaciones, lo que lo hace robusto a la diferencia de
escalas entre encoders. Se anota una salvedad honesta: esa robustez es decisiva
al fusionar rankers heterogéneos, y aquí ambos son densos con similitud coseno en
el mismo rango, de modo que **CombSUM** —también admitido por la §8.4— podría
rendir igual o mejor. La comparación entre ambos, y el valor de `k₀`, entran al
barrido contra el conjunto de evaluación interno.

**Post-filtros sobre metadata (§8.7).** La correspondencia consulta→fenómeno es
conocida (q001–q016 → F1, q017–q032 → F2, q033–q050 → F3), y el campo `fenomeno`
de la Tabla 1 permite filtrar por ella. Se aplica como **realce suave y no como
filtro duro**, porque varias consultas admiten evidencia transversal: q005 y q046
abordan Colombia desde fenómenos distintos, y q027 cruza inteligencia artificial
con operaciones espaciales. Un filtro estricto cerraría el acceso a documentos
legítimamente relevantes.

**Diversificación.** Se limita cuántas posiciones del top-10 puede ocupar un mismo
documento. Sin ese tope un solo documento puede ocupar las diez posiciones
evaluadas; si resulta no ser relevante, la consulta se pierde entera.

**Deduplicación.** 4.445 fragmentos del corpus (3%) tienen texto idéntico: las
mismas tablas reimpresas en informes distintos, los mismos encabezados
institucionales. Dos copias del mismo texto tienen la misma relevancia y gastan
dos de las diez posiciones que evalúa NDCG@10 para informar una sola vez. Se
eliminan de la lista de fragmentos, pero no de la agregación a documento: dos
documentos distintos pueden compartir un texto y descartar uno lo volvería
inalcanzable.

**Agregación al nivel de documento (§8.6).** Se emplea **max pooling**: cada
documento hereda la puntuación de su mejor fragmento. Se descarta la suma de
puntuaciones por su sesgo de longitud —un documento con cuarenta fragmentos
débiles de 0,15 acumula 6,0 y desplaza a un informe preciso que contiene la
respuesta con 0,85—. Dada la heterogeneidad de tamaños documentada en la sección
1, ese sesgo sería severo. El número de fragmentos recuperados actúa solo como
desempate, con peso suficientemente pequeño para no alterar el orden principal.

**Restricción sobre modelos generativos (§8.3).** Ningún modelo generativo
interviene en ninguna etapa: no hay reordenamiento por LLM, reformulación o
expansión de la consulta, filtrado generativo ni síntesis de fragmentos. Todas
las operaciones se realizan sobre vectores, puntuaciones de similitud y metadata.

---

## 8. Formato de salida y evaluación interna

### Construcción de la respuesta (§9.2)

Por consulta se devuelven los 3 documentos y los 10 fragmentos más relevantes. El
76,5% de los fragmentos del índice supera las 250 palabras, así que la regla de la
§9.2.1 se activa en la mayoría de las posiciones entregadas.

Esa sección admite dos tratamientos: dividir el fragmento en sub-fragmentos que
respeten el límite, cada uno con su propio rango, o recortarlo. **Se recorta,
respetando fronteras oracionales.** La decisión se tomó midiendo ambas
alternativas sobre el mismo índice: subdividir reduce el texto entregado de
103.791 a 83.971 palabras por corrida y la cobertura de 10 fragmentos distintos a
6, porque la segunda pieza de un fragmento de 321 palabras son 71 y ocupa un
rango entero con poco contenido. Recortar deja las diez posiciones con 250
palabras cada una.

La subdivisión queda implementada y desactivada: si el ground truth muestra que
la cola de un fragmento bien posicionado aporta más que un fragmento nuevo peor
posicionado, se activa desde el barrido. Cuando se activa, el `chunk_id`
reportado sigue siendo el del fragmento original del índice, compartido por todas
sus piezas —trazabilidad y no emparejamiento, §9.2.1 y §10.2.1—, y el tope por
documento se cuenta sobre posiciones entregadas para que la subdivisión no anule
la diversificación.

El archivo se valida contra el esquema de la §9.3 antes de la entrega:
exactamente 50 líneas en el orden q001–q050, 3 documentos y 10 fragmentos por
consulta, todos los campos de la Tabla 2 presentes y ningún fragmento por encima
del límite de palabras.

### Evaluación interna (§10.2)

El ground truth oficial no es público. Se anota uno propio sobre las 50 consultas
reales, lo que permite optimizar contra la distribución real de evaluación en
lugar de sustitutos sintéticos. El pool de anotación se construye con la unión
del top-N de cada encoder **por separado**, no con la salida fusionada: anotar
solo la salida final sesgaría la medición hacia lo que la configuración actual ya
recupera, y cualquier cambio que sacara a la superficie documentos nuevos
aparecería como ruido sin anotar.

**NDCG@10** y **F1@3** están implementadas como transcripción literal de las
fórmulas de la §10.2, incluyendo la normalización `mín(|D*q|, 3)` del recall, sin
la cual una consulta con cinco documentos relevantes quedaría topada en 0,6 aun
devolviendo los tres mejores posibles. Ambas están cubiertas por pruebas
automáticas contra valores derivados a mano, porque son las que decidirán cada
barrido: una métrica equivocada no se delataría sola.

---

## 9. Reproducibilidad

`generador.py` se entrega dentro de `entrega/`, junto al índice que lee y al
archivo de resultados que escribe, según la estructura de la §1.4. Carga los
índices persistidos, lee el archivo de consultas y regenera `resultados.jsonl`
sin reindexar el corpus.

Es un archivo autónomo: no importa nada del resto del proyecto, porque el
directorio de entrega debe funcionar por sí solo en manos del evaluador. Sus
únicas dependencias son las bibliotecas con las que se construyó el índice.

Las semillas están fijadas y las versiones de modelo ancladas. Una comprobación
automatizada contrasta la salida del entregable contra la del pipeline de
desarrollo sobre el mismo índice y exige que sean idénticas, de modo que las dos
implementaciones no puedan divergir en silencio.

La §1.4 excluye de la evaluación toda entrega que no reproduzca sus resultados;
se trata como criterio eliminatorio y no como recomendación.
