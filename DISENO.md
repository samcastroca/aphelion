# Diseño técnico — CODEFEST AD ASTRA 2026, Etapa 1

Base de conocimiento vectorial para recuperación multilingüe sobre el corpus ADL.
Este documento fija las decisiones de arquitectura y sirve de insumo directo para el
informe técnico entregable (máx. 8 páginas).

---

## 1. Caracterización del corpus

Medido sobre los 1826 archivos del inventario oficial (`Indice_Datos_Codefest.xlsx`),
verificado al 100% contra disco.

| Formato | Archivos | Texto extraíble |
|---|---:|---:|
| PDF | 760 | 87.5M caracteres (~21.9M tokens) |
| JSON | 964 | 9.5M caracteres (~2.4M tokens) |
| PBF | 73 | atributos de mapa |
| CSV | 26 | filas tabulares |
| XLSX | 6 | datasets AI Index |
| Imagen | 9 | OCR si aplica |
| **Total** | **1826** | **~97M caracteres (~24.3M tokens)** |

Estimación resultante: **~56.000 fragmentos** con chunks de 512 tokens y 15% de solape.

### Hechos que condicionan el diseño

1. **53 PDFs sin capa de texto.** 48 de ellos son `ALERTAS_informes*.pdf`, informes
   escaneados de la Defensoría del Pueblo (~1.000 páginas en total). Responden
   directamente a las preguntas q033–q050. Requieren OCR obligatoriamente.
2. **Asimetría de idioma.** Las 50 consultas están en español; el corpus es
   mayoritariamente inglés (CSET, SWF, Atlantic Council, SIPRI, UNOOSA, ESA, CSIS)
   con presencia de portugués (INPE). **La recuperación cross-lingual es el problema
   central**, no un caso borde.
3. **Heterogeneidad extrema de tamaño.** Alertas JSON de ~1.3k caracteres conviven con
   atlas RESDAL de cientos de páginas. La estrategia de chunking y la agregación a
   documento deben tolerar ambos extremos.
4. **59 nombres de archivo duplicados** (186 archivos) en carpetas distintas, con
   contenido distinto — verificado por hash MD5. No hay duplicados exactos en el corpus.
5. **Dos morfologías de JSON:** artículos individuales (`body_text`) y catálogos
   (listas de registros que describen otros archivos).

---

## 2. Identificación de documentos

**No se inventan identificadores.** Se adopta el esquema que ADL define en el Excel:

- `doc_id` = `{fenomeno}-{codigo_observatorio}-{consecutivo:03d}` → `F1-CSET-001`
- `fuente` = campo `Nombre estandarizado` del inventario, literal
- `fenomeno` = 1, 2 o 3, tomado del inventario (no inferido)
- `ruta` = campo adicional no obligatorio, para trazabilidad interna

**Justificación:** la §10.2.1 establece que el emparejamiento con el ground truth se
realiza por `fuente`, no por `doc_id`. Cualquier divergencia respecto al nombre
original de ADL invalida la métrica F1@3 con independencia de la calidad del sistema.

La ingesta itera sobre el inventario, **nunca sobre un listado del directorio**. Esto
excluye por construcción los artefactos que contaminarían el índice: `.DS_Store`,
los XLSX de índice y —crítico— `Extracto_Preguntas_50_v2.pdf`, cuya indexación
introduciría las consultas de evaluación dentro del corpus.

---

## 3. Extracción por formato

| Formato | Herramienta | Notas |
|---|---|---|
| PDF con texto | PyMuPDF | Orden de lectura por bloques, no por posición cruda |
| PDF escaneado | OCR (ver §4) | 53 archivos detectados por umbral <200 caracteres |
| JSON artículo | parser propio | `title` + `body_paragraphs`/`body_text` al cuerpo; `url`, `date`, `authors`, `tags` a metadata |
| JSON catálogo | parser propio | Cada registro de la lista es un fragmento independiente |
| CSV / XLSX | pandas | Cada fila es una unidad, con `columna: valor` como contexto |
| PBF | mapbox-vector-tile | **Un solo nivel de zoom**; atributos como `clave: valor` |
| Imagen | OCR | Descartar si es portada sin texto informativo |

### Limpieza y normalización

Aplicada uniformemente tras la extracción:

- Normalización a UTF-8 (NFC).
- Eliminación de caracteres de control y colapso de espacios redundantes.
- Supresión de boilerplate: numeración de página, encabezados y pies repetidos.
  Se detectan por frecuencia de línea a lo largo del documento, no por heurística fija.
- Detección de idioma predominante, almacenada como metadata para post-filtros.

---

## 4. OCR

**Decisión: evaluación comparativa antes de comprometerse.** Se comparan
`baidu/Unlimited-OCR` (3B, MIT, VLM con R-SWA) y Tesseract (`-l spa`) sobre una muestra
de los informes escaneados.

**Criterios de selección**, en orden de peso:

1. **Ausencia de alucinación.** Un VLM que inventa texto introduce evidencia falsa en
   el índice; Tesseract, al fallar, produce ruido evidente. Este criterio domina sobre
   la calidad promedio.
2. Fidelidad de diacríticos (tildes, ñ) — el corpus escaneado es español.
3. Legibilidad de tablas y estructura.
4. Velocidad — secundaria: son ~1.000 páginas y el proceso corre una sola vez.

**Encuadre normativo:** el OCR pertenece al preprocesamiento (§2.1), que la
especificación explícitamente recomienda. Las prohibiciones sobre arquitecturas decoder
aplican a la generación de embeddings (§4.2) y al módulo de recuperación (§8.3), etapas
en las que ningún modelo generativo interviene en este diseño. El uso de un VLM para
OCR se declara explícitamente en el informe técnico.

---

## 5. Chunking

**Configuración base: 512 tokens, 15% de solape, corte en frontera oracional.**

- La segmentación en oraciones usa `pysbd`, con soporte nativo para español, inglés y
  portugués.
- El conteo de tokens usa el tokenizador del propio encoder, no una aproximación por
  palabras.
- Cuando el límite cae dentro de una oración, el corte retrocede al final de la última
  oración completa que quepa. **Ninguna oración cruza la frontera entre fragmentos**
  (requisito §3.3).
- **Documentos cortos** (alertas, registros de catálogo, filas tabulares) se emiten como
  fragmento único sin forzar el tamaño objetivo.

**Justificación.** Evaluaciones comparativas sobre PDFs técnicos sitúan el chunking
recursivo de 512 tokens con solape alineado a oraciones como el mejor compromiso entre
las dos granularidades evaluadas (F1 a nivel fragmento 0.92, a nivel documento 0.86).
Se descarta explícitamente el chunking semántico: pese a su buen desempeño a nivel
fragmento (0.91), colapsa a nivel documento (0.42) al producir fragmentos de ~43 tokens
que fragmentan el contexto — precisamente la métrica F1@3 que se evalúa.

**Parámetro sujeto a barrido experimental.** El tamaño de chunk presenta tensión entre
las dos métricas: 512 tokens favorece NDCG@10 mientras 1024 favorece F1@3. Dado que el
Conteo de Borda pondera ambas por igual, el valor se fija empíricamente mediante barrido
sobre {384, 512, 768} contra el conjunto de evaluación interno (§9).

### Metadata por fragmento

Campos obligatorios (Tabla 1 de la especificación) más extensiones propias:

| Campo | Origen |
|---|---|
| `doc_id`, `fuente`, `fenomeno` | Inventario ADL |
| `chunk_id` | `{doc_id}-chunk-{posicion:04d}` |
| `formato`, `posicion`, `num_tokens`, `texto` | Derivados de la ingesta |
| `idioma`, `observatorio`, `ruta`, `titulo`, `fecha` | Extensiones para post-filtros |

---

## 6. Codificación semántica

**Encoders seleccionados:**

| Modelo | Licencia | Arquitectura | Dim. | Contexto | Rol |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | MIT | XLM-RoBERTa (encoder) | 1024 | 8192 | Principal |
| `intfloat/multilingual-e5-large` | MIT | XLM-RoBERTa (encoder) | 1024 | 512 | Complementario |

**Justificación de BGE-M3 como principal:** supera a mE5-large en recuperación en
español (0.727 frente a 0.660 en MIRACL-VISION), su ventana de 8192 tokens elimina
restricciones sobre el chunking, y produce representaciones densas y sparse en una única
pasada, aportando sensibilidad léxica sin índice adicional.

**Justificación del segundo encoder:** mE5-large mantiene un espacio vectorial más
particionado por idioma, lo que reduce el sesgo de recuperar documentos en un idioma
distinto al de la consulta. Complementa la debilidad conocida de BGE-M3 en esa dimensión.

**Modelos descartados y por qué:**

- **Qwen3-Embedding** (líder en MTEB multilingüe, 70.88): arquitectura decoder derivada
  de un backbone autoregresivo. **Prohibido explícitamente por la §4.2.**
- **Jina Embeddings v3**: licencia CC-BY-NC-4.0, incompatible con el criterio de
  licencia de la §4.3.
- **LaBSE**: entrenado con objetivo de alineación de frases paralelas; carece de noción
  de relevancia asimétrica y rinde 18.80 en recuperación zero-shot.

---

## 7. Índice vectorial

**`IndexFlatIP` con vectores normalizados a norma unitaria**, un índice independiente
por encoder.

Con ~56k vectores, la búsqueda exhaustiva es exacta y se resuelve en milisegundos. Los
índices aproximados (IVF, HNSW) intercambian exactitud por una velocidad que este
volumen no requiere, y en esta tarea la exactitud del ranking constituye la métrica.

La normalización previa hace que el producto interno sea equivalente a la similitud
coseno (§8.2).

**Persistencia:** `faiss.write_index()` produce `index.faiss`; la metadata se serializa
en `metadata.jsonl` **preservando el orden de inserción**, de modo que el identificador
interno de FAISS coincide con el número de línea.

---

## 8. Módulo de recuperación

Flujo por consulta:

1. La consulta se codifica con **el mismo encoder** usado en la indexación, con el
   prefijo de instrucción que cada modelo requiere (`query: ` para E5).
2. Cada índice devuelve sus `k` candidatos con sus puntuaciones.
3. **Fusión por Reciprocal Rank Fusion**: `score(c) = Σ 1/(k₀ + rank_j(c))`.
   RRF opera sobre posiciones y no sobre puntuaciones, lo que lo hace inmune a la
   diferencia de escalas entre espacios vectoriales distintos. `k₀ = 60` como punto de
   partida, ajustado empíricamente.
4. **Diversificación**: tope de fragmentos por documento en el top-10, para evitar que
   un único documento erróneo consuma las diez posiciones evaluadas.
5. **Agregación a documento por max pooling**: la puntuación de un documento es la de su
   mejor fragmento. Se descarta sum pooling por su sesgo de longitud — un documento con
   40 fragmentos débiles (0.15) acumula 6.0 y desplaza a un informe preciso con un
   fragmento de 0.85. Dada la heterogeneidad de tamaños del corpus (§1.3), este sesgo
   sería severo.
6. **Boost suave por fenómeno**, no filtro duro. La correspondencia consulta→fenómeno es
   conocida (q001–q016 → F1, q017–q032 → F2, q033–q050 → F3), pero varias consultas
   admiten evidencia transversal (q005 y q046 mencionan Colombia desde fenómenos
   distintos; q027 cruza IA con operaciones espaciales). Un filtro estricto cerraría el
   acceso a documentos legítimamente relevantes.

**Ningún modelo generativo interviene** en ninguna etapa: no hay reranking por LLM,
expansión de consulta, filtrado generativo ni síntesis. Todas las operaciones se
realizan sobre vectores, puntuaciones de similitud y metadata (§8.3).

### Construcción de la salida

- **3 documentos** por consulta, ordenados por puntuación agregada.
- **10 fragmentos** por consulta, cada uno de **máximo 250 palabras**. Los fragmentos que
  exceden el límite se subdividen respetando fronteras oracionales; los cortos pueden
  concatenarse con fragmentos adyacentes del mismo documento. En ambos casos el
  `chunk_id` reportado es el del fragmento original del índice (§9.2.1).

---

## 9. Evaluación interna

El ground truth oficial no es público. Se construye uno propio sobre **las 50 consultas
reales** (`Extracto_Preguntas_50_v2.pdf`), lo que permite optimizar contra la
distribución real de evaluación en lugar de consultas sintéticas.

**Procedimiento:**

1. Generar un pool de candidatos por consulta uniendo el top-20 de **dos configuraciones
   distintas** (BGE-M3 y mE5 por separado). La unión mitiga el sesgo de evaluar
   únicamente lo que el sistema ya encuentra.
2. Etiquetado manual en tres niveles: relevante / parcialmente relevante / no relevante.
3. Reparto entre los cuatro integrantes, con solapamiento parcial para medir acuerdo.

**Métricas implementadas** (réplica exacta de las fórmulas de la §10.2): NDCG@10 sobre
fragmentos y F1@3 sobre documentos, con la normalización `min(|Dq|, 3)` en el recall.

Este conjunto es la base para el barrido de hiperparámetros: tamaño de chunk, solape,
`k₀` de RRF, umbral de diversificación e intensidad del boost por fenómeno.

---

## 10. Grafo de conocimiento (bonus)

Componente opcional, planificado **al final** del cronograma y sacrificable si el tiempo
aprieta. NER multilingüe sobre los fragmentos, extracción de relaciones por
dependencias sintácticas, y construcción con NetworkX exportando a `grafo.graphml`. Cada
tripleta conserva referencia a su `doc_id` y `chunk_id` de origen. Se integra a la
recuperación como una lista ordenada adicional dentro del RRF.

---

## 11. Estructura de la entrega

```
entrega/
  resultados.jsonl            50 líneas, q001–q050
  generador.py                reproduce resultados.jsonl desde el índice
  informe_tecnico.pdf         máx. 8 páginas
  base_vectorial/
    encoder_bge-m3/
      index.faiss
      metadata.jsonl
    encoder_me5-large/
      index.faiss
      metadata.jsonl
    grafo/
      grafo.graphml           si aplica
```

**Reproducibilidad.** `generador.py` carga los índices persistidos, lee el archivo de
consultas y regenera `resultados.jsonl` sin reindexar. Semillas fijadas y versiones de
modelo ancladas. La especificación excluye de la evaluación las entregas que no
reproducen sus resultados; se trata como requisito eliminatorio, no como recomendación.

---

## 12. Riesgos identificados

| Riesgo | Mitigación |
|---|---|
| Alucinación del OCR sobre escaneos degradados | Comparación previa contra Tesseract; el criterio de fidelidad domina sobre el de calidad promedio |
| Colisión de nombres en `fuente` (59 casos) | Se reporta el nombre estandarizado literal; la ruta se conserva aparte para trazabilidad |
| Sesgo del ground truth interno hacia lo que el sistema ya recupera | Pool generado por unión de dos configuraciones distintas |
| Tensión NDCG@10 / F1@3 en el tamaño de chunk | Barrido empírico sobre el conjunto interno, no elección a priori |
| Entorno: PyTorch no soporta Python 3.14 | Proyecto fijado a Python 3.12 mediante `uv` |
| Indexación en máquina sin GPU | Pipeline agnóstico de dispositivo, reanudable por lotes |
