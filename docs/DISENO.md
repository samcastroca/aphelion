# Diseño técnico — CODEFEST AD ASTRA 2026, Etapa 1

Base de conocimiento vectorial para recuperación multilingüe sobre el corpus ADL.
Este documento fija las decisiones de arquitectura y sirve de insumo directo para el
informe técnico entregable (máx. 8 páginas).

---

## 1. Caracterización del corpus

Medido sobre los 1826 archivos del inventario oficial (`Indice_Datos_Codefest.xlsx`),
verificado al 100% contra disco.

| Formato | Archivos | Texto extraído |
|---|---:|---:|
| PDF | 759 | 131.6M caracteres |
| CSV | 26 | 111.9M caracteres |
| PBF | 73 | 7.3M caracteres (atributos de mapa) |
| JSON | 954 | 4.7M caracteres |
| XLSX | 6 | 1.4M caracteres |
| Imagen / TXT | 9 | OCR / texto plano |
| **Total** | **1826** | **~257M caracteres** |

Resultado medido tras ejecutar la extracción y la fragmentación: **149.571
fragmentos, 72.3M tokens, mediana de 499 tokens**, sobre 1758 de los 1826
documentos. Los CSV aportan 90.442 fragmentos (60% del índice), casi todos
listados bibliográficos de PubMed ajenos a las consultas.

> La estimación previa a la medición era de ~56.000 fragmentos y ~97M
> caracteres. Se quedó corta en un factor de 2.7 porque subestimó el volumen
> tabular de los CSV. Las cifras de arriba son las medidas, no las estimadas.

### Hechos que condicionan el diseño

1. **60 PDFs sin capa de texto.** 48 de ellos son `ALERTAS_informes*.pdf`, informes
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

**Justificación:** el emparejamiento con el ground truth se
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
| PDF escaneado | OCR (ver más abajo) | 60 archivos detectados por umbral <200 caracteres |
| JSON artículo | parser propio | `title` + `body_paragraphs`/`body_text` al cuerpo; `url`, `date`, `authors`, `tags` a metadata |
| JSON catálogo | parser propio | Cada registro de la lista es un fragmento independiente |
| CSV / XLSX | pandas | Cada fila es una unidad, con `columna: valor` como contexto |
| PBF | mapbox-vector-tile | Atributos como `clave: valor`, deduplicados **dentro** de cada tesela |
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

**Decisión: Tesseract (`-l spa`) a 200 dpi.** Se descartó `baidu/Unlimited-OCR`
(3B, MIT, VLM con R-SWA), pese a su mejor manejo de layout, por dos razones en
este orden:

1. **Es una arquitectura decoder, y el reto las prohíbe en la construcción del
   índice.** El OCR pertenece al preprocesamiento, donde el reto lo recomienda
   explícitamente, y el argumento de que la prohibición no lo alcanza es
   defendible. Pero el texto que produce termina indexado, y la sanción por una
   lectura estricta es la exclusión, no una penalización. El riesgo asimétrico
   decide: no hay ganancia de calidad que compense quedar fuera de la evaluación.
2. **Ausencia de alucinación.** Un VLM que inventa texto introduce evidencia falsa
   en el índice, que puede acabar presentada al jurado como respuesta. Tesseract,
   al fallar, produce ruido evidente.

**Control de calidad.** Como un OCR que falla en silencio es peor que uno que
revienta, sobre cada documento se miden tres señales y se revisa manualmente todo
lo que dispare alguna: densidad de diacríticos (un texto español sin tildes ni
eñes indica el paquete de idioma equivocado), proporción de caracteres no
imprimibles y palabras por página.

---

## 5. Chunking

**Configuración base: 512 tokens, 15% de solape, corte en frontera oracional.**

- La segmentación en oraciones usa `pysbd`, con soporte nativo para español, inglés y
  portugués.
- El conteo de tokens usa el tokenizador del propio encoder, no una aproximación por
  palabras.
- Cuando el límite cae dentro de una oración, el corte retrocede al final de la última
  oración completa que quepa: **ninguna oración cruza la frontera entre fragmentos**,
  que es un requisito del reto.
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
sobre {384, 512, 768} contra el conjunto de evaluación interno.

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
  de un backbone autoregresivo. **El reto prohíbe explícitamente los decoders.**
- **Jina Embeddings v3**: licencia CC-BY-NC-4.0, incompatible con el criterio de
  licencia.
- **LaBSE**: entrenado con objetivo de alineación de frases paralelas; carece de noción
  de relevancia asimétrica y rinde 18.80 en recuperación zero-shot.

---

## 7. Índice vectorial

**`IndexFlatIP` con vectores normalizados a norma unitaria**, un índice independiente
por encoder.

Con ~150k vectores, la búsqueda exhaustiva es exacta y se resuelve en milisegundos. Los
índices aproximados (IVF, HNSW) intercambian exactitud por una velocidad que este
volumen no requiere, y en esta tarea la exactitud del ranking constituye la métrica.

La normalización previa hace que el producto interno sea equivalente a la similitud
coseno.

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
   fragmento de 0.85. Dada la heterogeneidad de tamaños del corpus, este sesgo
   sería severo.
6. **Boost suave por fenómeno**, no filtro duro. La correspondencia consulta→fenómeno es
   conocida (q001–q016 → F1, q017–q032 → F2, q033–q050 → F3), pero varias consultas
   admiten evidencia transversal (q005 y q046 mencionan Colombia desde fenómenos
   distintos; q027 cruza IA con operaciones espaciales). Un filtro estricto cerraría el
   acceso a documentos legítimamente relevantes.

**Ningún modelo generativo interviene** en ninguna etapa: no hay reranking por LLM,
expansión de consulta, filtrado generativo ni síntesis. Todas las operaciones se
realizan sobre vectores, puntuaciones de similitud y metadata.

### Construcción de la salida

- **3 documentos** por consulta, ordenados por puntuación agregada.
- **10 fragmentos** por consulta, cada uno de **máximo 250 palabras**. Los fragmentos que
  exceden el límite se subdividen respetando fronteras oracionales; los cortos pueden
  concatenarse con fragmentos adyacentes del mismo documento. En ambos casos el
  `chunk_id` reportado es el del fragmento original del índice.

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

**Métricas implementadas**, transcritas literalmente de las fórmulas del reto: NDCG@10 sobre
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
  generador.py                código fuente versionado; el resto es generado
  resultados.jsonl            50 líneas, q001–q050
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

De esos cuatro elementos, `generador.py` es el único que es código fuente: vive
versionado en `entrega/` porque es donde el reto lo exige, y hay una sola copia.
Los otros tres son artefactos que produce `scripts/etapas/05_empaquetar.py` y no se
versionan.

**Reproducibilidad.** `generador.py` carga los índices persistidos, lee el archivo de
consultas y regenera `resultados.jsonl` sin reindexar. Semillas fijadas y versiones de
modelo ancladas. La especificación excluye de la evaluación las entregas que no
reproducen sus resultados; se trata como requisito eliminatorio, no como recomendación.

**Por eso `generador.py` es autónomo.** El jurado recibe solo `entrega/`: un
script que importe `src/aphelion` no arranca en sus manos, y eso basta para
quedar excluido. El entregable es un único archivo sin dependencias del proyecto,
escrito en inglés, cuyas únicas importaciones son las bibliotecas con las que se
construyó el índice.

El precio es tener la política de recuperación escrita dos veces:
`aphelion.recuperacion` para iterar durante el desarrollo, y el entregable.
`scripts/etapas/06_verificar.py` corre ambas sobre el mismo índice y exige
salidas idénticas, para que no puedan divergir en silencio. `05_empaquetar.py`
ejecuta el generador desde dentro de `entrega/`, de modo que lo que se valida es
el entregable resolviendo sus rutas como lo hará el evaluador.

---

## 12. Contraste con la literatura

Revisión de 39 fuentes sobre recuperación densa multilingüe (agosto 2026). Lo que
confirma el diseño, lo que lo contradice y lo que revela como oportunidad.

**Confirmado.** BGE-M3 supera a mE5-large en recuperación cross-lingual: 67,8
frente a 65,4 nDCG@10 en MIRACL sobre 18 idiomas. La elección de encoder
principal está bien fundada. 512 tokens es el valor por defecto defendible: un
barrido de 2026 sobre siete estrategias lo situó primero, y un estudio de
LlamaIndex sitúa el pico de fidelidad en 1024 — el rango 512–1024 es el
razonable, y nuestra tensión NDCG/F1 es real.

**La deduplicación tiene respaldo.** El filtrado de fragmentos redundantes reduce
el índice entre un 25% y un 36% con caídas de recall inferiores al 6%, y la
deduplicación byte-exacta antes de ensamblar el contexto no degrada la calidad de
salida de forma medible. El cambio implementado va en la dirección correcta.

**Contradicho: el solape puede no servir para nada.** Un análisis sistemático de
enero de 2026 encontró que el solape entre fragmentos **no aporta beneficio
medible** y solo incrementa el coste de indexación. Nuestro 15% cuesta
aproximadamente un 15% más de vectores y de tiempo de codificación. Debe entrar
en el barrido como candidato a eliminarse, no darse por bueno.

**Debilitado: la justificación de RRF.** Se argumentó que RRF es inmune a la
diferencia de escalas entre espacios vectoriales. Ese argumento vale para fusionar
BM25 con recuperación densa, donde las escalas son incomparables. Aquí fusionamos
**dos encoders densos**, ambos con similitud coseno en el mismo rango: el problema
que RRF resuelve apenas existe. Bruch et al. (2022) muestran que la combinación
convexa de puntuaciones normalizadas **supera a RRF en dominio** cuando hay
etiquetas de relevancia disponibles. En cuanto exista el ground truth interno,
CombSUM normalizado debe compararse contra RRF en lugar de asumirlo. El valor
`k₀ = 60` proviene de corpus a escala TREC; para colecciones menores se recomienda
entre 10 y 20, así que también entra al barrido.

**Oportunidad pendiente de medir: la cabeza sparse de BGE-M3.** El modelo produce
representaciones densas, sparse y multi-vector en una sola pasada. En MIRACL, la
cabeza densa sola rinde 67,8 y la combinación de las tres alcanza **70,0**. Usar
las tres no infringe la prohibición: ninguna es un decoder.

El matiz que impide adoptarla a ciegas: **la recuperación sparse rinde peor
justamente en cross-lingual**, donde el solapamiento de vocabulario entre consulta
y documento es mínimo — que es la situación dominante de este corpus, con 130.090
fragmentos en inglés frente a consultas en español. En cambio destaca en
documentos largos, donde las palabras clave discriminan mejor que la similitud
densa. Cabe esperar que ayude en las siglas y topónimos que sí cruzan idiomas
(NBQR, ASAT, Chocó, Arauca) y estorbe en el resto. Sin ground truth no se puede
saber cuál efecto domina, así que entra al barrido, no al diseño.

Receta de implementación, ya verificada contra la documentación del modelo:

1. Los pesos están en `sparse_linear.pt` del repositorio `BAAI/bge-m3`: una
   `Linear(1024 → 1)`, aparte del backbone. No es la cabeza `MaskedLM` que usa
   SPLADE, de ahí que las implementaciones que asumen SPLADE fallen buscando
   `lm_head.decoder`.
2. Peso por token: `w = ReLU(sparse_linear(last_hidden_state))`.
3. Descartar tokens especiales (`CLS`, `SEP`, `PAD`, `UNK`) y pesos ≤ 0.
4. Si un mismo token aparece varias veces, conservar **solo su peso máximo**.
5. Puntuación léxica entre consulta y pasaje: `s_lex = Σ_{t ∈ q∩p} w_qt · w_pt`.
6. Fusión sugerida por los autores para las tres cabezas: `[0.4, 0.2, 0.4]`
   (densa, sparse, ColBERT).

El coste de almacenamiento es moderado —solo se guardan los tokens presentes en
cada texto— frente a ColBERT, que multiplica el índice por más de diez y queda
descartado por volumen.

### Codificación en hardware AMD

La máquina de desarrollo tiene una Radeon RX 6650 XT (gfx1032, RDNA2, 8 GB) y un
Ryzen 5 3400G de cuatro núcleos. De las cuatro rutas posibles a esa GPU, tres
están cerradas:

| Ruta | Estado |
|---|---|
| CUDA | No aplica: la GPU es AMD |
| ROCm en Windows o WSL2 | gfx1032 no está en la matriz de soporte, y AMD declinó admitir `HSA_OVERRIDE_GFX_VERSION` en Windows |
| `torch-directml` | Degradaría torch de 2.13 a 2.4 |
| **ONNX Runtime + DirectML** | **Funciona.** 5,0 frag/s frente a 0,27 en CPU |

Se descarta `optimum` como intermediario: exige `transformers<5` y el proyecto
está en 5.14.1. La exportación con `torch.onnx.export` no necesita esa capa.

**El pooling se hornea dentro del grafo ONNX.** BGE-M3 usa el token CLS y E5
promedia con la máscara de atención. Confundirlos no produce ningún error: baja
el coseno contra la referencia de 0,999999 a 0,81, una degradación que solo
aparecería como recuperación mediocre. Al generarlo desde
`config.ENCODERS[...]["pooling"]`, el lado de Python no puede equivocarse.

---

## 13. Riesgos identificados

| Riesgo | Mitigación |
|---|---|
| El entregable no arranca fuera del repositorio → exclusión | `generador.py` vive en `entrega/` y es autónomo; `05_empaquetar.py` lo ejecuta desde ahí, con las rutas que verá el jurado |
| El OCR por VLM cae bajo la prohibición de decoders → exclusión | Se usa Tesseract; el VLM queda descartado y la decisión se declara en el informe |
| OCR que falla en silencio e inyecta ruido en el índice | Tres señales por documento (diacríticos, no imprimibles, palabras/página) y revisión manual de lo que las dispare |
| Caché de embeddings reutilizada entre corridas distintas → índice desalineado | La ruta de caché lleva la huella del archivo de fragmentos; además se verifica el tamaño de cada bloque al cargarlo |
| Colisión de nombres en `fuente` (59 casos) | Se reporta el nombre estandarizado literal; la ruta se conserva aparte para trazabilidad |
| Sesgo del ground truth interno hacia lo que el sistema ya recupera | Pool generado por unión de dos configuraciones distintas |
| Tensión NDCG@10 / F1@3 en el tamaño de chunk | Barrido empírico sobre el conjunto interno, no elección a priori |
| Indexación sin GPU: 0,27 frag/s en el Ryzen 5 3400G, 65 h por encoder | **Resuelto.** ONNX Runtime sobre DirectML aprovecha la Radeon RX 6650 XT: 5,0 frag/s, unas 3,5 h por encoder |
| El backend ONNX podría divergir del PyTorch con el que el jurado codifica las consultas | `onnx_dml.verificar()` compara ambos antes de indexar y aborta si el coseno baja de 0,999. Medido: 0,99974 |
