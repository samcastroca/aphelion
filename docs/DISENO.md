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

**Imágenes sueltas.** Las imágenes del corpus (JPG, PNG, AVIF) se entregan a
Tesseract abiertas con Pillow, no rasterizadas con PyMuPDF: PyMuPDF no abre
AVIF, y el corpus trae uno (`F2-SWF-065`). Pillow ≥ 11.2 abre los cuatro
formatos, así que las imágenes van todas por el mismo camino y PyMuPDF queda
solo para los PDF.

---

## 5. Chunking

**Configuración base: ventana de 512 tokens, 15% de solape, corte en frontera
oracional.**

**El presupuesto de fragmentación es 504, no 512.** La ventana de mE5-large
incluye lo que el fragmento no trae: los tokens especiales (`<s>`, `</s>`) y el
prefijo `passage: ` con que E5 exige codificar los pasajes. Un fragmento
presupuestado a 512 tokens de contenido entra al modelo con ~518 y pierde la
cola en silencio — sin error, solo con vectores que no representan el final del
texto. `config.RESERVA_TOKENS_ENCODER = 8` descuenta ese sobrecoste del
presupuesto (`CHUNK_PRESUPUESTO = 504`); la ventana del modelo y del grafo ONNX
sigue siendo 512.

- La segmentación en oraciones usa `pysbd`, con reglas propias para español e inglés.
  **Portugués no está entre los idiomas que soporta** —pedírselo levanta un
  `ValueError`— así que esos documentos se segmentan con el modelo inglés. El fallo
  era silencioso: la llamada vive dentro de un `try`, y los 7.617 fragmentos en
  portugués caían al `except`, que trata el párrafo entero como una sola oración.
  Sus cortes no caían entonces en frontera oracional sino en la ventana de tokens
  de último recurso. Corregido en `_IDIOMAS_PYSBD`; **surte efecto al reindexar**.
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
| `idioma`, `observatorio`, `ruta`, `titulo` | Extensiones para post-filtros |

Los nombres de los ocho obligatorios son los que fija la Tabla 1, literalmente, y
no se traducen: renombrarlos equivale a no emitirlos. Los cuatro añadidos sí son
libres —la §3.4 los autoriza y los ejemplifica en español—, y van en español por
coherencia con los otros ocho. `tests/test_metadata.py` fija ambas cosas.

**Sobre `texto` "sin modificaciones".** La Tabla 1 lo describe así, mientras la
§2.2 exige limpieza y normalización: las dos cosas no pueden cumplirse a la letra
a la vez. Se interpreta que prohíbe reescribir o resumir el fragmento, no
normalizarlo, porque de lo contrario la §2.2 no tendría contenido posible. Lo que
se aplica es conservador y reversible en significado: NFC, ligaduras tipográficas
que PyMuPDF conserva (`ﬁ` → `fi`), caracteres de control, espacios redundantes,
encabezados repetidos detectados por frecuencia y numeración de página. No se
pasa a minúsculas, no se quitan tildes y no se toca ninguna palabra del cuerpo.

---

## 6. Codificación semántica

**Encoders seleccionados:**

| Modelo | Licencia | Arquitectura | Dim. | Contexto | Rol |
|---|---|---|---|---|---|
| `BAAI/bge-m3` | MIT | XLM-RoBERTa (encoder) | 1024 | 8192 | Principal |
| `intfloat/multilingual-e5-large` | MIT | XLM-RoBERTa (encoder) | 1024 | 512 | Complementario — retirado tras medirlo |

**Justificación de BGE-M3 como principal:** supera a mE5-large en recuperación en
español (0.727 frente a 0.660 en MIRACL-VISION), su ventana de 8192 tokens elimina
restricciones sobre el chunking, y produce representaciones densas y sparse en una única
pasada, aportando sensibilidad léxica sin índice adicional.

**El segundo encoder se retiró tras medirlo.** El argumento era que mE5-large
mantiene un espacio vectorial más particionado por idioma y compensa la debilidad
conocida de BGE-M3 ahí. Sobre el corpus completo y el entregable real no se
sostiene: BGE-M3 solo, con agregación top2, gana +0,1074 de NDCG@10 frente a la
fusión de los dos con max pooling (p = 0,0002, permutación pareada sobre las 50
consultas) y cede 0,0213 de F1@3, que no se distingue del ruido (p = 0,51).

La receta `familias` descarta además la explicación alternativa: si la fusión
valiera por decorrelación de errores, sustituir mE5 por GTE —otra familia de
preentrenamiento— debería mejorarla, y no lo hace. La fusión aportaba poco, y una
agregación mejor recupera lo que aportaba a la mitad de coste: un índice en vez de
dos y la mitad del tiempo de codificación.

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
5. **Agregación a documento por top2, con la `fuente` como clave**: la puntuación
   de un documento es la media de sus dos mejores fragmentos. Se descarta sum
   pooling por su sesgo de longitud — un documento con 40 fragmentos débiles (0.15)
   acumula 6.0 y desplaza a un informe preciso con un fragmento de 0.85. Dada la
   heterogeneidad de tamaños del corpus, este sesgo sería severo.

   Frente a max pooling, top2 premia que la evidencia esté en varios fragmentos
   buenos sin dejar que cuarenta flojos ganen por acumulación. Medido, recupera el
   F1@3 del fenómeno 2 —donde max más perdía— de 0,7708 a 0,8750.

   La agregación agrupa por `fuente` y no por `doc_id`, porque el jurado empareja
   los documentos por `fuente` (§10.2.1) y el corpus tiene 59 nombres
   estandarizados repetidos en 186 archivos: dos `doc_id` con la misma fuente en
   el top-3 cuentan como un solo acierto posible y desperdician una posición. De
   cada fuente se reporta el `doc_id` de su mejor fragmento.
6. **Boost suave por fenómeno**, no filtro duro. La correspondencia consulta→fenómeno es
   conocida (q001–q016 → F1, q017–q032 → F2, q033–q050 → F3), pero varias consultas
   admiten evidencia transversal (q005 y q046 mencionan Colombia desde fenómenos
   distintos; q027 cruza IA con operaciones espaciales). Un filtro estricto cerraría el
   acceso a documentos legítimamente relevantes.
7. **Umbral relativo de similitud** (post-filtro de la §8.7, desactivado por
   defecto). Descarta de cada índice los candidatos por debajo de una fracción de
   la mejor similitud de esa consulta en ese índice. Es relativo y no absoluto
   porque las escalas de coseno de los dos encoders no son comparables — mE5
   comprime todo hacia 0,75–0,92 —, y como el ranking viene ordenado, equivale a
   un k adaptativo por consulta que nunca reordena lo que conserva. El valor se
   decide en el barrido (`umbral` en la rejilla), no a priori: su efecto depende
   de cuánta cola de ruido traiga cada consulta, que sin ground truth no se sabe.

### Construcción de la salida y relleno

El recorte a 250 palabras y la subdivisión segmentan con **el idioma del
fragmento** (campo `idioma` de la metadata), no con español fijo: el 87% del
corpus está en inglés y las abreviaturas que `pysbd` reconoce dependen del
idioma; segmentar mal mueve el punto de corte.

El tope de fragmentos por documento es firme mientras haya alternativa y cede
cuando no la hay: si la primera pasada deja posiciones vacías —pool corto, tope
alcanzado—, una segunda las rellena con piezas aún no emitidas ignorando el
tope. Un fragmento real de un documento ya representado informa más que la
alternativa, que era repetir literalmente el último fragmento para completar el
esquema. La repetición queda como último recurso, solo cuando el pool entero se
agota. Para que esa segunda pasada tenga repuestos, la diversificación entrega
el triple de candidatos de los que se publican.

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

## 9.bis Ciclo de experimentación

Comparar configuraciones sobre el corpus completo no cabe en el tiempo
disponible: codificar sus 64.484 fragmentos cuesta horas **por encoder**. El
ciclo de mejora corre sobre una submuestra y separa lo caro de lo barato.

**La submuestra** (`data/submuestra.json`, versionada) tiene tres capas, en este
orden de prioridad:

1. **Todo lo que el ground truth toca** —265 documentos— y va entero. Si faltara
   un documento juzgado, sus juicios se volverían ceros y la métrica mentiría
   hacia abajo; si faltara uno marcado relevante, el F1@3 de esa consulta tendría
   techo artificialmente bajo.
2. **Los negativos difíciles**: documentos que los encoders puntúan alto sin ser
   relevantes, tomados de los rankings profundos cacheados. Un subconjunto de
   material relevante más ruido aleatorio es *más fácil* que el corpus real, y una
   configuración puede verse bien ahí solo porque no compite contra nada.
3. **Relleno estratificado** por (fenómeno, formato), para no distorsionar la
   mezcla de idiomas y tipos de archivo.

Resultado: 353 documentos, 28.557 fragmentos, cobertura 1383/1383 de los juicios,
**2,3 veces menos** que el corpus. Los documentos **no se recortan**, y eso pone
el suelo en ese 44%: el barrido re-fragmenta y necesita el texto completo, y una
de las cosas que compara es la agregación a documento —max frente a suma— cuyo
resultado depende de la longitud real. Recortar los largos sesgaría esa
comparación a favor de la suma.

**El emparejamiento por texto es obligatorio, no una comodidad.** El ground truth
se anotó sobre chunks de 504 tokens; al probar 256 o 768 sus `chunk_id` dejan de
existir y evaluar por identificador daría ceros disfrazados de mediciones.
`evaluacion/emparejamiento.py` decide la relevancia por solape de n-gramas de 5
palabras con contención en el sentido más favorable —para que emparejen tanto el
chunk grande que contiene al juzgado como el pequeño contenido en él—. Es además
lo que hará el jurado (§10.2.1).

El umbral está **calibrado por las dos caras**, no elegido a ojo:

| umbral | juicios recuperables tras re-fragmentar | NDCG@10 frente al emparejamiento por `chunk_id` |
|---|---:|---:|
| 0,60 | 97,0% (los 29 huérfanos entre 0,588 y 0,599) | +0,002 |
| **0,55** | **99,9%** | **+0,005** |

El sesgo positivo es una décima parte del intervalo de confianza, así que bajar el
umbral recupera emparejamientos legítimos sin inventar relevancia. Cambiarlo exige
repetir las dos medidas.

**Un experimento prueba lo que se le pide.** Cada dimensión se elige por separado
y lo que no se diga toma el valor de la entrega, así que pedir un encoder, un
chunking, una fusión y una agregación es *una* corrida. El producto completo son
decenas de miles y sobreajusta: con 50 consultas, elegir el máximo de cientos de
configuraciones sobre el mismo conjunto da una ganadora que no se sostiene fuera.
Los presets abren **una** dimensión y dejan el resto fija, que es como se lee un
resultado sin confundir el efecto de una cosa con el de otra.

**Las recetas** (`evaluacion/recetas.py`) son el otro extremo: configuraciones
completas, con los doce parámetros fijados, que compiten por entrar a la entrega.
Una receta es una corrida y trae escrito en qué se basa, de modo que el catálogo
no es una lista de corazonadas. Nueve, ordenadas para que las que comparten la
fragmentación de la entrega vayan primero:

| receta | qué cambia | de dónde sale la apuesta |
|---|---|---|
| `entrega` | nada; es la vara de medir | sin ella se compara entre candidatas, que es la pregunta equivocada |
| `dos-encoders` | la entrega anterior: bge-m3 + mE5, max | retirar un encoder es lo más caro de revertir; se conserva medible |
| `convexa` | fusión convexa normalizada | RRF tira la magnitud; Bruch et al. 2022 la conserva |
| `familias` | bge-m3 + gte en vez de + mE5 | los dos que se fusionaban salen de XLM-R y se equivocan igual |
| `filtrado` | umbral relativo 0,9 | §8.7, implementado y desactivado esperando este número |
| `barato` | mE5-base solo | si la brecha cabe en el intervalo, el modelo grande no se paga |
| `sin-recorte` | 345 tokens | el 79,9% de los fragmentos excede las 250 palabras y se entrega truncado |
| `granular` | 256 tokens, sin solape | más precisión por vector, y ningún fragmento llega al límite |
| `contexto` | 768 tokens, top3 | la apuesta contraria, sobre la otra métrica |

Cada una se informa como diferencia contra `entrega`, y la mitad del ancho de su
intervalo marca el suelo por debajo del cual la diferencia no distingue nada. La
receta `entrega` se comprueba en las pruebas contra `config`: si se desincroniza,
todas las comparaciones se harían contra algo que no se entrega.

Los fragmentos y los índices se comparten entre experimentos, porque su contenido
queda determinado por (chunking, encoder); las métricas van por experimento. Los
pares (chunking, encoder) a construir salen de los planes y no del producto de las
opciones: `contexto` pide 768 tokens **solo** con BGE-M3, y codificar ahí mE5 sería
una hora de GPU en un índice truncado que ningún plan va a consultar.

**Truncamiento silencioso.** Codificar un fragmento más largo que la ventana del
modelo no falla: el tokenizador lo corta y devuelve un vector que no representa la
cola. Es el peor fallo posible en un barrido, porque produce números plausibles de
una configuración que nadie está midiendo. `config.cabe_en_ventana` lo comprueba
contra `max_tokens` más la reserva, las recetas lo tienen como invariante de
prueba, y la rejilla avisa sin abortar —el resto de sus pares sigue siendo
informativo—.

**El catálogo de encoders** pasa de dos a siete. `ENCODERS_ENTREGA` son los que se
indexan para entregar; los otros cinco existen para el barrido y todos son
arquitecturas encoder con licencia permisiva, como exigen la §4.2 y la §4.3:
`me5-base` y `me5-small` por baratos —hacen viable barrer el chunking—,
`me5-large-instruct` por su asimetría instruida, `gte-multilingual-base` porque es
de otra familia y por eso es el que más consenso *nuevo* puede aportar a la
fusión, y `labse` como **control**: el informe lo descarta por argumento tomado de
la literatura, y medirlo convierte esa afirmación en un número propio.

---

## 10. Grafo de conocimiento (bonus)

Componente opcional. Las tres etapas de la §7.2 viven en `src/aphelion/grafo/` y las
construye `scripts/etapas/04_grafo.py`, que comparte número con `04_indexar` porque
comparte escalón: el grafo se arma sobre los fragmentos, no sobre los vectores.

**Se construye pero no se conecta.** Son dos decisiones distintas y atarlas sale caro.
La §7 puntúa por *construir* el grafo y la §1.4 lo recoge como archivo; la §8.5, en
cambio, dice que el equipo «puede» combinarlo con los resultados vectoriales. Entregar
`base_vectorial/grafo/grafo.graphml` cobra el bonus sin tocar nada. Meter un canal en la
política de recuperación obliga a escribirla también dentro de `generador.py`, a ampliar
la prueba de paridad y a revalidar `06_verificar` — es decir, a poner en juego lo único
eliminatorio del reto a cambio de una ganancia que nadie ha medido todavía. Se mide
primero, en el barrido; se conecta después, si sale a favor.

**Etapa 1 — NER.** Multilingüe y de tipos abiertos: las entidades que importan a estas
50 consultas son *sistema de armas autónomo*, *órbita baja terrestre* o *política
pública*, y un clasificador con las cuatro clases de CoNLL las mete todas en MISC. Se usa
`fastino/gliner2-multi-v1` (Apache-2.0, declara es/en/pt) por su export ONNX, que corre
sobre el mismo DirectML que ya monta la codificación. Quedan fuera
`Babelscape/wikineural-multilingual-ner` —CC BY-NC-SA 4.0, y es el multilingüe que uno
elegiría por defecto— y los modelos de spaCy en español, que son GPL-3.0 por herencia de
UD AnCora. El paquete `gliner` v1 no se usa aunque su modelo sirva: declara
`transformers<5.14.0` contra el `>=5.14.1` del proyecto, el mismo choque que dejó fuera a
`optimum`.

**Etapa 2 — relaciones por patrones.** La §7.2 admite tres vías y las otras dos salen
caras: los extractores generativos (REBEL, mREBEL) son decoders, que la §4.2 y la §8.3
prohíben, y mREBEL es además CC BY-NC-SA 4.0; GLiREL, que sí clasifica en lugar de
generar, es también no comercial; y los parsers de dependencias de spaCy arrastran GPL en
español y CC BY-SA en portugués. Queda un inventario cerrado de catorce patrones sobre
pares de menciones vecinas dentro de la misma oración, con las pasivas comprobadas antes
que las activas para que «desarrollado por» no salga con el sujeto y el objeto
intercambiados. Las conjugaciones están enumeradas y no comodinadas: `desarroll\w+`
captura también «desarrollo», que es un sustantivo, e inventaría relaciones que el texto
no afirma. `knowledgator/gliner-relex-large-v1.0` (Apache-2.0, clasificación sobre pares)
entra al barrido como candidato a mejorarlo, no como sustituto: no declara soporte
multilingüe.

**Etapa 3 — construcción.** Tres tipos de nodo y tres de arista:

```
documento --CONTIENE--> fragmento --MENCIONA--> entidad
                                   entidad --RELACION--> entidad
```

La procedencia (`doc_id`, `chunk_id`, evidencia textual) va en la arista `RELACION`, que
es lo que exige la §7.2 y de paso resuelve la serialización: el escritor GraphML de
NetworkX solo admite `int`, `str`, `float` y `bool`, así que guardar «los chunks donde
aparece esta entidad» como lista dentro del nodo levanta un `TypeError` al exportar —al
final de la etapa más cara, no al construirla—. `construccion.validar_escalares` lo
comprueba antes de escribir y `tests/test_grafo.py` exporta y relee el grafo en cada
`pytest`.

La poda es parte del diseño y no una optimización posterior: GraphML es XML y medio
millón de aristas de mención se van a cientos de megas. Entran las entidades que aparecen
en dos documentos o más —una entidad citada una sola vez no conecta nada— y en menos del
5% del corpus —por encima es membrete institucional—, con un tope de doce por fragmento.
El tope superior nunca baja del inferior: sobre una muestra pequeña, el 5% redondea a
cero y la ventana se queda sin nadie dentro, que es un grafo vacío sin ningún error.

**La canonicalización es el paso que la especificación no nombra** y el que decide si
esto informa o es ruido. Sin unificar superficies, «Estados Unidos», «United States» y
«EE. UU.» son tres nodos. La normalización resuelve mayúsculas, tildes, puntuación y
artículos de los tres idiomas; el cruce entre idiomas lo cierra agrupar los nombres por
su embedding **con el mismo encoder que construye el índice**, que es cross-lingüe por
construcción, ya está cargado en esa etapa y no añade ninguna licencia que declarar.

**Si se conecta**, es como una lista ordenada más dentro del RRF:
`recuperacion.fusionar_rrf` recibe un diccionario de canal a ranking, así que el grafo
entra como una entrada más y la fusión no se entera. Tiene que ser RRF y no la convexa ni
CombSUM: este canal puntúa contando menciones y relaciones, no cosenos, y solo RRF es
inmune a esa diferencia de escala. Iría dentro de `Recuperador.buscar` y no de `ordenar`,
porque depende de la consulta y no de los hiperparámetros de fusión: es la mitad que el
barrido cachea una vez para reordenar cien.

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

**Resuelto: la justificación de RRF ya no aplica a la entrega.** Se argumentó que
RRF es inmune a la diferencia de escalas entre espacios vectoriales. Ese argumento
vale para fusionar BM25 con recuperación densa, donde las escalas son
incomparables; fusionando **dos encoders densos** con coseno en el mismo rango, el
problema que RRF resuelve apenas existía. La discusión quedó zanjada por la vía
corta: la entrega usa un solo índice, así que no hay nada que fusionar. Lo que
sigue vale para el barrido, donde la fusión se compara. Bruch et al. (2022) muestran que la combinación
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
| Truncamiento silencioso en mE5: la ventana de 512 incluye especiales y el prefijo `passage: ` que el fragmento no trae | El presupuesto de fragmentación reserva 8 tokens (`CHUNK_PRESUPUESTO = 504`); la ventana del modelo sigue en 512 |
| Dos `doc_id` con la misma `fuente` en el top-3 cuentan como un solo acierto (59 nombres repetidos) | La agregación a documento agrupa por `fuente` y reporta el `doc_id` del mejor fragmento |
| El relleno del esquema repetía fragmentos, que no aportan ni a NDCG ni a F1 | Segunda pasada con piezas reales no emitidas; la repetición queda como último recurso |
| Divergencia entre el paquete y `generador.py` descubierta tras horas de GPU | `tests/test_paridad_entregable.py` compara ambos sobre un índice sintético en cada `pytest`; `06_verificar.py` lo repite sobre el índice real |
| Un merge deshace en `generador.py` mejoras que sí siguen en `src/` | Las dos comprobaciones de la fila anterior lo detectan; ya ocurrió una vez y así se encontró |
| `06_verificar.py` mantiene los encoders de las dos implementaciones a la vez y agota una GPU de 4 GB | El lado del paquete libera sus modelos y vacía la caché de CUDA antes de que arranque el del entregable |
| Idioma no soportado por `pysbd` con fallo silencioso dentro de un `try` | Solo `es` e `en` van a `pysbd`; el resto usa el segmentador inglés, y el mapa es idéntico en el paquete y en el entregable |
| Indexación sin GPU: 0,27 frag/s en el Ryzen 5 3400G, 65 h por encoder | **Resuelto.** ONNX Runtime sobre DirectML aprovecha la Radeon RX 6650 XT: 5,0 frag/s, unas 3,5 h por encoder |
| El backend ONNX podría divergir del PyTorch con el que el jurado codifica las consultas | `onnx_dml.verificar()` compara ambos antes de indexar y aborta si el coseno baja de 0,999. Medido: 0,99974 |
