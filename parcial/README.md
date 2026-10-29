### CC0F4 - Examen Parcial
#### Del prototipo LLM y la recuperación densa a un sistema RAG evaluable

**Curso:** CC0F4 - Tópicos de Ciencia de la Computación I · **Periodo:** 2026-2
**Estudiante:** César Augusto Sánchez Malaspina (modalidad individual)
**Tema (registrado en PC1):** Agente de revisión científica que identifica desacuerdos entre artículos mediante citation context y generación estructurada
**Commit evaluado:** ver la sección [12](#12-reproducibilidad-y-commit-evaluado)

---

#### 1. Problema y continuidad

Investigadores y revisores necesitan saber si dos líneas de trabajo **coinciden o discrepan** sobre una afirmación (por ejemplo, «el cómputo óptimo favorece modelos muy grandes con pocos datos») y **con qué evidencia**. Una respuesta fluida no basta: debe citar el pasaje exacto que la sustenta, o abstenerse.

| Etapa | Qué existía | Qué hace este parcial |
|---|---|---|
| **PC1** (`../src/`, `../data/`) | Qwen2.5-0.5B clasifica `agree/disagree/neutral` con JSON validado (Pydantic). El contexto se escribía a mano en 8 casos sintéticos | Reutiliza el modelo, `RelationEnum` y los tres niveles de validez (raw / recovered / schema-valid), ahora con contexto **recuperado** de artículos reales |
| **Línea base de recuperación** (alcance PC2) | — | Recuperación densa E5 + FAISS con qrels propios. Es el **sistema A** del experimento 1 (§6 del cuaderno) |
| **Parcial** (`parcial/`) | — | Corpus real de arXiv, 20 consultas + qrels, comparación A/B/C, RAG con citas a `passage_id`, herramienta con contratos |

**Pregunta experimental.** Con corpus, consultas, qrels, segmentación, modelos y decoding fijos: ¿qué aporta cada etapa de recuperación (léxica + fusión, reranking) a la evidencia recuperada, y cuánto cambian la corrección y la fidelidad de la respuesta del mismo LLM cuando recibe esa evidencia?

**Correcciones a PC1** (la evidencia previa no se modificó):
1. **El decoding «greedy» de PC1 no era greedy puro.** El `generation_config.json` de Qwen2.5 aplica `repetition_penalty=1.1` incluso con `do_sample=False`, y `../src/evaluator.py` no lo sobrescribía. Lo detecté al comparar el argmax de los logits con la salida de `generate()` (sección 8.2 y §9 del cuaderno). Aquí se fija `repetition_penalty=1.0`.
2. La conclusión 2 de PC1 («greedy supera consistentemente al sampling») no está respaldada por su propia tabla: ambos obtienen 50 % de accuracy y 100 % de schema-valid.
3. La conclusión 3 de PC1 («el contexto ruidoso degrada el accuracy un 25 %») contradice su tabla: C0 = C1 = 50 %.
4. Los 8 casos de PC1 eran sintéticos. El examen exige datos reales (sección 3).

Fuera de alcance, según el enunciado: agentes, ReAct, memoria, interfaz gráfica y despliegue.

---

#### 2. Cómo ejecutar

Todo el examen está en un único cuaderno, **[`notebooks/Examen_Parcial_CC0F4.ipynb`](notebooks/Examen_Parcial_CC0F4.ipynb)**, con el mismo estilo de los Cuadernos del curso: secciones numeradas, explicación, código, verificaciones `assert` y tablas. Se entrega **ejecutado**, con todas sus salidas.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r parcial/requirements.txt
.venv/bin/python -m ipykernel install --user --name cc0f4-parcial --display-name "Python (cc0f4-parcial)"
cd parcial/notebooks && ../../.venv/bin/jupyter lab Examen_Parcial_CC0F4.ipynb   # Run All (~5 min en CPU)
```

Ejecución sin interfaz (regenera `results/`):

```bash
cd parcial/notebooks && ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cc0f4-parcial Examen_Parcial_CC0F4.ipynb
```

- Los metadatos de arXiv ya están descargados en `data/raw/arxiv_raw.json`; el cuaderno no necesita internet.
- La primera celda fija `OMP_NUM_THREADS=1`: en macOS, faiss-cpu y torch cargan runtimes OpenMP distintos y el proceso terminaba con segfault (código 139). Con un solo hilo, además, las latencias de las etapas son comparables.
- La sección 11 es la **demostración** de la defensa: cambia `DEMO_QID` para probar otra consulta.

---

#### 3. Corpus, consultas y qrels

Detalle completo en [`data/README.md`](data/README.md).

| Elemento | Valor |
|---|---|
| Documentos | **36 artículos reales de arXiv** (2017-2024) sobre 12 debates de ML/NLP |
| Passages | **135** (unidad de relevancia), con ID estable `arXivID-Pnn` |
| Chunks | 110 (`target_words=100`, `overlap=1 passage`) |
| Consultas | **20** de recuperación (10 claras, 3 ambiguas, 5 difíciles, 2 con distractor) + 2 sin respuesta para abstención |
| Qrels | 56 juicios graduados (2 = directa, 1 = parcial), **anotados antes de ejecutar los sistemas**; 2,8 passages relevantes por consulta en promedio |
| Licencia | Metadatos y abstracts de arXiv: CC0 1.0 |

Cada consulta es un objeto `StructuredQuery` (§2 del cuaderno) que se valida antes de recuperar: ID, texto, traducción al español, `topic` restringido al dominio del corpus (validación de dominio), afirmación bajo prueba y dificultad.

---

#### 4. Arquitectura

```text
consulta estructurada (Pydantic: StructuredQuery)
   ├─> E5 "query:" + FAISS IndexFlatIP ──────────┐            A = denso
   └─> BM25Okapi (k1=1.5, b=0.75) ───────────────┴─> RRF k=60   B = A + BM25 + RRF
                                                      └─> Cross-Encoder mMiniLM   C = B + reranker
                                                           └─> top-3 chunks -> passages [ID]
                                                                └─> Qwen2.5-0.5B, greedy -> JSON {relation, answer, citations, confidence}
                                                                     └─> validación: parse -> schema -> citas ⊆ contexto

herramienta independiente: ToolCall JSON -> allowlist -> contrato -> dominio -> ejecución (retry/timeout) -> contrato de salida
```

| Sección del cuaderno | Función | Origen en el curso |
|---|---|---|
| 1. Configuración | Todos los parámetros, fijados antes de ejecutar | Cuaderno5 (celda 2) |
| 2. Contratos | `StructuredQuery`, `RAGAnswer`, validación semántica | Cuaderno3, PC1 `schema.py` |
| 3. Corpus | Carga de arXiv, segmentación en passages, `validate_benchmark` | Cuaderno5 (celda 5), `Semana4/datos` |
| 4. Segmentación | `build_chunks` con overlap | Cuaderno5 (celda 7) |
| 5. Métricas | Recall@k, RR, nDCG graduado, bootstrap, tipos de fallo, citas | Cuaderno7 (celdas 8-27) |
| 6. Recuperadores | E5+FAISS, BM25, RRF, Cross-Encoder, latencia por etapa | Cuaderno4, Cuaderno5 (celdas 9-20) |
| 7. Experimento 1 | A/B/C, bootstrap, análisis por consulta | Cuaderno5, Cuaderno7 |
| 8. Experimento 2 | Contexto con IDs, generación, validación, auditoría manual | Cuaderno5 (`generate_with_context`), Cuaderno7 |
| 9. Inspección del modelo | Arquitectura, máscara causal, logits, KV cache | Cuaderno1, Cuaderno2 |
| 10. Experimento 3 | Herramienta, `ToolResult`/`ToolError`, política de retry | Cuaderno6 (celdas 5-13, ejercicios 1-6) |
| 11. Demostración | Consulta → evidencia → respuesta con citas → métrica → error de herramienta | — |

Los prompts están en `prompts/` (versionados, con SHA-256 en `results/exp2/run_config.json`).

---

#### 5. Matriz de trazabilidad (Semanas 1-7)

| Sem. | Recurso revisado | Concepto | Dónde se aplica | Evidencia verificable | Limitación o conclusión propia |
|---|---|---|---|---|---|
| 1 | `Cuaderno1`, `Lectura2` (Attention Is All You Need) | Tokens, embeddings, Q/K/V, atención con máscara causal | §9 | `results/model_inspection.json`: 24 capas, 14 HEADs Q / **2 HEADs K/V (GQA)**, head_dim 64. «Chinchilla» se tokeniza en 3 sub-tokens; atención máxima sobre la diagonal = **0.0**; filas suman 1 | Qwen usa GQA, la misma idea que el corpus discute en Q09 (MQA/GQA): el modelo ejemplifica el trade-off calidad/memoria que se evalúa |
| 2 | `Cuaderno2`, `Lectura3` (logits, KV cache) | Logits, softmax, greedy vs. sampling, ventana de contexto, KV cache | §1, §8, §9 | P(primer token `{"`) = 0,77 en Q17. KV cache = **24 KB/token** en fp32, 7 veces menos que con MHA; la secuencia más larga (1014 tokens) usa 24,9 MB de 32 768 posibles. Se verifica que el prompt cabe en la ventana | Se eligió greedy por estabilidad del JSON. **Hallazgo:** `generate()` aplicaba `repetition_penalty=1.1` oculto, que penaliza tokens del prompt (como `{"` del contrato) y cambió el primer token a un bloque de código |
| 3 | `Cuaderno3` | Instrucción / contexto / contrato, JSON Schema, validación estructural vs. semántica | §2, §8 (`validate_output`), `prompts/` | Schema-valid: closed-book 25 %, RAG 83 % (`results/exp2/auto_summary.csv`). Las respuestas `"Yes"` fallan `min_length=5`; las citas fuera del contexto se marcan como error semántico | Un JSON válido no implica una respuesta correcta: RAG tiene 83 % schema-valid pero solo 25 % de relación correcta |
| 4 | `Cuaderno4`, `Laboratorio4` | Chunking, overlap, embeddings E5 (`query:`/`passage:`), FAISS, top-k, qrels | §4 (`build_chunks`), §6 (`dense_ranking`) | 110 chunks; el máximo es de 234 tokens, por debajo del límite de 512 del encoder. IndexFlatIP exacto, dim 384 (`results/exp1/run_config.json`) | Con `target_words=180` (línea base de S4), 20 de 36 abstracts quedarían en un único chunk; se usa la variante de 100 palabras. Qrels por passage, no por chunk |
| 5 | `Cuaderno5`, `Lectura5` | BM25, híbrido, RRF, Cross-Encoder, RAG | §6, §7 | Tabla A/B/C (sección 7). El reranker gana en consultas difíciles (nDCG 0,49 → 0,63) y cuesta ~253 ms por consulta | A→B mezcla dos cambios: se agrega BM25-only como diagnóstico para separarlos. No hay superioridad general |
| 6 | `Cuaderno6`, `Lectura6` | Function calling, contratos, validación de dominio, errores, retries, timeouts | §10 | 12 de 12 casos coinciden con lo esperado; las pruebas de la política se verifican con `assert` (`results/exp3/`) | Reintentar solo `transient`/`timeout`; un veredicto negativo no es un error |
| 7 | `Cuaderno7`, `Lectura7` | Recall@k, MRR, nDCG, grounding, citas, latencia, análisis de errores | §5, §7, §8, `results/exp2/manual_audit.json` | Bootstrap pareado (5000 remuestreos), tipos de fallo, auditoría manual de 26 afirmaciones (sin LLM como juez) | Recall@1 está acotado a 0,44 por los múltiples relevantes; con 20 consultas los IC95 son anchos |

---

#### 6. Registro experimental: resumen

| Exp. | Pregunta | Línea base | Modificación | Métrica principal | Resultado | Conclusión |
|---|---|---|---|---|---|---|
| 1 | ¿Qué aporta cada etapa de recuperación? | A: E5 + FAISS | B: +BM25+RRF · C: +Cross-Encoder | nDCG@5, Recall@k, MRR, latencia | nDCG@5: 0,634 / 0,631 / 0,671; C cuesta unas 25 veces la latencia de B | El reranker ayuda en consultas difíciles; ningún IC95 de nDCG excluye 0 |
| 2 | ¿La evidencia mejora la respuesta del mismo LLM? | Closed-book | RAG con top-3 de C | Corrección, citas y faithfulness (manual) | Corrección 0,08 → 0,25; faithfulness 0,14 → 0,63; citas 0,21 | La evidencia llega, pero un modelo de 0,5B no la usa con fidelidad |
| 3 | ¿Los contratos hacen verificable la herramienta? | — | Política E del Cuaderno6 | Caso esperado = observado | 12/12 | Retry selectivo y errores tipados |

---

#### 7. Experimento 1 - Recuperación y reranking

**Variables fijas:** corpus `parcial-v1`, 20 consultas, qrels, chunks (100 palabras, overlap 1), E5-small, mMiniLM-L12, `candidate_depth=10`, RRF `k=60`, BM25 `k1=1.5, b=0.75, ε=0.25`, CPU con 1 hilo, warm-up excluido. Las métricas se calculan a nivel de **passage** después de deduplicar el overlap (`expand_to_passage_ranking`).

| Sistema | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Latencia media (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| BM25 solo (diagnóstico) | 0,246 | 0,571 | 0,725 | 0,735 | 0,615 | 0,2 | 0,3 |
| **A** denso | 0,221 | **0,642** | 0,763 | 0,750 | 0,634 | 10,4 | 11,8 |
| **B** A + BM25 + RRF | **0,263** | 0,567 | 0,754 | **0,785** | 0,631 | 10,6 | 12,0 |
| **C** B + Cross-Encoder | 0,221 | 0,617 | **0,825** | 0,758 | **0,671** | 263,3 | 333,6 |

**Latencia por etapa** (media / p95, en ms): denso 10,4 / 11,8 · BM25 0,18 / 0,26 · RRF 0,02 / 0,03 · **reranker 252,6 / 323,6**. Las etapas se ejecutan en secuencia; la latencia del sistema B es la suma de denso, BM25 y RRF. Excluye la carga de modelos y la construcción del índice.

**Bootstrap pareado sobre consultas** (`results/exp1/bootstrap_paired.csv`):

| Comparación | Métrica | Δ medio | IC95 | Gana / empata / pierde |
|---|---|---|---|---|
| B - A | Recall@3 | -0,075 | [-0,150, -0,004] | 1 / 14 / 5 |
| B - A | MRR | +0,035 | [-0,080, +0,158] | 4 / 13 / 3 |
| B - A | nDCG@5 | -0,003 | [-0,091, +0,082] | 7 / 6 / 7 |
| C - B | Recall@3 | +0,050 | [-0,013, +0,125] | 3 / 16 / 1 |
| C - B | MRR | -0,027 | [-0,167, +0,112] | 2 / 15 / 3 |
| C - B | nDCG@5 | +0,040 | [-0,049, +0,134] | 5 / 10 / 5 |

**nDCG@5 por dificultad:** difíciles A 0,465 → B 0,486 → **C 0,634**; ambiguas A **0,737** → B 0,660 → C 0,670; claras y con distractor, prácticamente iguales.

**Interpretación**

- **A frente a B cambia dos cosas a la vez:** agrega una señal léxica y una fusión por posiciones. Por eso se incluye BM25 solo como diagnóstico. BM25 solo tiene un nDCG@5 inferior a A (0,615 frente a 0,634), pero rescata consultas donde el término exacto importa. La fusión mejora MRR (el primer relevante sube) y empeora Recall@3, cuyo IC95 es el único que excluye 0: en 5 consultas, los candidatos léxicos desplazan a un segundo passage relevante.
- **B frente a C aísla el reranker:** mismo conjunto de candidatos, solo cambia el orden. Mejora Recall@5 (0,754 → 0,825) y las consultas difíciles, a cambio de multiplicar la latencia por unas 25 veces (263 ms frente a 10,6 ms).
- **No se declara un sistema superior en general:** todos los IC95 de nDCG@5 y MRR cruzan 0, y los empates dominan.

**Consultas con diferencias relevantes** (`results/exp1/rankings.jsonl`):

- **Q04** (¿la explicación CoT es fiel?): A = 0,845 → **B = 0,325** → **C = 0,912**. BM25 coincide con «chain-of-thought» y sube a rank 1 el abstract de Wei et al. (2201.11903-C001, no relevante). RRF lo combina con el ranking denso y desplaza a Turpin et al. El Cross-Encoder, que lee la consulta y el passage juntos, devuelve a rank 1 el chunk de Turpin con los passages P04 (grado 2) y P05 (grado 1). Es el patrón «coincidencia léxica engañosa».
- **Q11** (¿un modelo pequeño con RLHF se prefiere a uno grande?): A = 0,631 → **B = 1,000** → **C = 0,500**. BM25 acierta con «human feedback» + «smaller» y B pone 2203.02155-P03 en rank 1. El reranker prefiere el passage de DPO (2305.18290-C001), que habla de «human preferences» y RLHF pero no responde la pregunta. Es un **resultado negativo** del reranker: relevancia temática no es lo mismo que evidencia.
- **Q15** (¿la atención aproximada logra speedup real?): A = 0,363 → B = 0,820. El denso pone primero un passage de FlashAttention sobre atención block-sparse «faster than any existing approximate attention method», que es un distractor semántico. BM25 recupera P01 («often do not achieve wall-clock speedup»).

**Tipos de fallo** (`results/exp1/failure_types.csv`, top-3): las tres condiciones tienen 14 consultas en `PARTIAL_RECALL`. Ninguna tiene `RETRIEVAL_MISS` dentro de los 10 candidatos. B tiene 1 `RANKING_ERROR` (Q13: el único passage directo, 2402.16775-P04, cae fuera del top-3 y el CE lo recupera en rank 1). BM25 solo tiene un `RETRIEVAL_MISS` (Q13).

**Error:** en Q01, B baja de 0,425 a 0,185. 2001.08361-P03 (Kaplan: «very large models on a relatively modest amount of data», grado 2) queda fuera del top-3 en las tres condiciones, y eso afecta al experimento 2.

**Limitaciones.** Hay 20 consultas y un solo anotador; Recall@1 tiene una cota máxima de 0,4375, porque hay 2,8 relevantes por consulta en promedio; la latencia se midió en CPU con un hilo, con BM25 y denso en secuencia; no se ajustaron `k`, la profundidad ni la segmentación después de observar resultados.

**Conclusión.** Para este corpus de abstracts, la recuperación densa ya cubre la mayor parte de la evidencia en el top-10. El reranker mejora el orden en las consultas difíciles, que son las que contienen dos posturas, y es la configuración que se declara para RAG (C). Su costo y sus fallos (Q11) quedan documentados.

---

#### 8. Experimento 2 - RAG y evidencia

**Pregunta:** ¿cambian la corrección y la fidelidad de la respuesta del mismo LLM cuando recibe la evidencia recuperada?

**Variables fijas:** `Qwen/Qwen2.5-0.5B-Instruct` (revisión `7ae5576`), fp32 en CPU, **greedy** (`do_sample=False`, `repetition_penalty=1.0`), `max_new_tokens=220`, las mismas 12 preguntas fijadas antes de ejecutar (`RAG_QUERY_IDS`, §1: 10 con respuesta y 2 donde corresponde abstenerse), el mismo contrato JSON.
**Modificación:** el prompt RAG incluye los passages de los top-3 chunks de la **condición C** con su `passage_id`. El closed-book recibe solo la pregunta.

##### 8.1 Resultados

| Modo | Schema-valid | Relación correcta | Corrección* | Citas* | Faithfulness* | Abstención correcta | ms/respuesta |
|---|---|---|---|---|---|---|---|
| Closed-book | 0,25 | 0,00 | 0,08 | — | 0,14 (1/7) | 0/2 | 1334 |
| **RAG** | **0,83** | **0,25** | **0,25** | 0,21 | **0,63 (12/19)** | 0/2 | 4884 |

\* Auditoría manual (`results/exp2/manual_audit.json`, protocolo incluido en el archivo). *Corrección*: correct = 1, partial = 0,5, incorrect = 0, frente a `reference_answers.jsonl`. *Citas*: 1 si todas las citas respaldan su afirmación, 0,5 si alguna respalda y ninguna contradice, 0 si alguna contradice o no hay citas. *Faithfulness*: afirmaciones atómicas sustentadas por el contexto (en RAG) o por el corpus (en closed-book). Las métricas automáticas complementarias (`auto_summary.csv`) muestran que el 100 % de las citas de RAG está dentro del contexto, pero solo el **38,5 %** apunta a un passage relevante según los qrels.

**Diagnóstico clave:** en las 10 preguntas con respuesta, **la evidencia gold estuvo en el contexto** (`gold_in_context` en `generations.jsonl`). El cuello de botella es la generación, no la recuperación.

##### 8.2 Errores y respuestas no sustentadas

- **Q17, cita que contradice la respuesta.** RAG responde «The dense retriever outperforms BM25, especially in zero-shot settings» y cita 2104.08663-P03 (BEIR), que dice que BM25 es un baseline robusto y que los modelos densos «often underperform» en zero-shot. La primera mitad está respaldada por DPR (2004.04906-P02, en dominio); la segunda está **contradicha** por su propia cita.
- **Q21 y Q22, falla de abstención (0 de 2).** Con Mamba, que no está en el corpus, RAG afirma «Mamba outperforms Transformers» y cita 5 passages irrelevantes (Linformer, GPTQ, Transformer). Con Llama 3, el distractor léxico «Chinchilla» lleva a responder sobre el cómputo de Chinchilla.
- **Q07 closed-book, alucinación bibliográfica.** Inventa «Zhang, Y., & Li, Y. (2020)… Journal of Artificial Intelligence Research, 64, 1-44».
- **Sesgo de etiqueta.** RAG responde `agree` en **12 de 12** casos (en el `raw_output`, incluidas las 2 respuestas que no cumplen el schema). El modelo interpreta `agree` como «sí a la pregunta», no como «los artículos coinciden». Es un problema de diseño de la etiqueta que se arrastra de PC1, donde las dos afirmaciones venían explícitas.
- **Citas en bloque.** El modelo copia casi todos los IDs del contexto. Por eso la métrica automática «cita dentro del contexto» (100 %) no sirve sola como evidencia de grounding.

**Mejor caso:** Q09 (MQA/GQA). Sus 4 afirmaciones están sustentadas («much faster to decode», «only minor quality degradation»…), pero no cita 1911.02150-P03, que es la fuente de dos de ellas.

**Corrida previa conservada** (`results/exp2_run0_repetition_penalty/`): con el `repetition_penalty=1.1` por defecto, RAG obtuvo 0,92 de schema-valid y la misma relación correcta (0,25). La inspección de logits (§9) mostró que el argmax del primer token era `{"` (p = 0,77), mientras que `generate()` producía un bloque de código json: la penalización actuaba sobre tokens que aparecen en el contrato del prompt. Se corrigió para que el decoding declarado («greedy») sea el ejecutado, y se re-ejecutó una sola vez con las mismas preguntas. La auditoría manual corresponde a la corrida definitiva.

**Limitaciones.** 12 preguntas; un solo modelo de 0,5B; auditoría de un solo anotador; las respuestas inválidas por schema se auditaron igualmente por contenido; no se probó otro prompt para no ajustar sobre el conjunto de evaluación.

**Conclusión.** RAG aumenta la validez del schema, la corrección y la fidelidad (de 0,14 a 0,63) con respecto a closed-book, pero no produce respuestas confiables con este modelo: la relación sigue siendo incorrecta en 9 de 12 casos, ninguna abstención es correcta y las citas no garantizan respaldo. La evaluación separada de corrección, citas y faithfulness muestra fallos que una métrica de fluidez o de «citas dentro del contexto» ocultaría.

---

#### 9. Experimento 3 - Herramienta con contratos

**Herramienta:** `citation.check_temporal_validity(citing_id, cited_id)`. Verifica con el catálogo local (fechas de arXiv) si una cita es **temporalmente posible**: un artículo no puede citar a otro publicado después que él. Reemplaza de forma determinista el campo `citation_valid` que en PC1 decidía el LLM.

| Contrato | Regla |
|---|---|
| Entrada | `citing_id`, `cited_id` con patrón `^\d{4}\.\d{4,5}$`; `extra="forbid"` |
| Dominio | Ambos existen en el catálogo; no hay autocita |
| Salida | `citing_date`, `cited_date`, `days_between`, `temporally_valid`, con el invariante `temporally_valid == (cited_date <= citing_date)` |
| Errores | `ToolError{code, kind, retryable, message}` con `kind ∈ {protocol, validation, domain, transient, timeout, execution, output_contract}` |
| Política | Hasta 3 intentos; backoff de 20 y 40 ms; timeout de 100 ms por intento; deadline total de 400 ms |

| Caso | Situación | Esperado | Observado | Intentos | ms |
|---|---|---|---|---|---|
| T01 | Chinchilla (2022) cita a Kaplan (2020) | ok (`true`) | ok | 1 | 0,56 |
| T02 | Kaplan «cita» a Chinchilla | ok (`false`) | ok | 1 | 0,02 |
| T03 | ID con formato inválido | validation | validation | 0 | 0,08 |
| T04 | Campo extra `year` | validation | validation | 0 | 0,01 |
| T05 | Artículo fuera del catálogo | domain | domain | 0 | 0,01 |
| T06 | Autocita | domain | domain | 0 | 0,01 |
| T07 | Fallo transitorio una vez | ok | ok | 2 | 21,2 |
| T08 | Timeout una vez | ok | ok | 2 | 122,4 |
| T09 | Timeout persistente | timeout | timeout | 3 | 365,7 |
| T10 | Salida que viola el contrato | output_contract | output_contract | 1 | 0,14 |
| T11 | Herramienta fuera de la allowlist | protocol | protocol | 0 | 0,02 |
| T12 | ToolCall mal formada | protocol | protocol | 0 | 0,05 |

**Cuándo reintentar.** Solo cuando `retryable=True`, es decir, en errores `transient` y `timeout` cuya causa es externa y puede desaparecer. Además debe quedar presupuesto en el deadline para el backoff (`can_backoff`). **No se reintentan** `validation`, `domain`, `protocol` ni `output_contract`: repetir los mismos argumentos produce el mismo error. La excepción se clasifica por **tipo**, nunca por el texto del mensaje.

**Error documentado.** T02 muestra que un resultado negativo (`temporally_valid=false`) **no es un error**: la herramienta funcionó y el hallazgo es de dominio. Confundirlo con `ToolError` haría reintentar inútilmente.

**Limitaciones.** El catálogo contiene solo los 36 artículos del corpus; los fallos se inyectan con `fault` (que no forma parte de la interfaz pública); la fecha usada es la de la primera versión de arXiv, no la de la publicación en un venue.

**Conclusión.** Los contratos convierten una decisión que en PC1 era probabilística en una verificación determinista, observable (con trazas por intento) y temporalmente acotada.

---

#### 10. Limitaciones generales

1. **Tamaño del benchmark:** 36 documentos, 135 passages y 20 consultas permiten una evaluación docente, pero **no prueban generalización estadística**.
2. **Un solo anotador** para los qrels y la auditoría, sin medida de acuerdo.
3. **Solo abstracts** en inglés; algunos hallazgos aparecen en un único passage.
4. **Un solo LLM pequeño** (0,5B). Los resultados del experimento 2 describen este modelo, no RAG en general.
5. **Latencias** medidas en un solo equipo, en CPU con 1 hilo.

#### 11. Conclusiones

1. La recuperación no fue el cuello de botella: con la condición C, la evidencia gold llegó al contexto en 10 de 10 preguntas. El reranker mejora las consultas difíciles con dos posturas (nDCG@5 0,49 → 0,63) a unas 25 veces la latencia, y falla en Q11.
2. A→B no aísla el aporte léxico. El diagnóstico BM25-only muestra que lo léxico ayuda cuando el término exacto importa (Q15) y daña cuando el término es ambiguo (Q04).
3. RAG mejora la fidelidad de 0,14 a 0,63, pero Qwen 0,5B no razona sobre posturas opuestas: responde `agree`, cita en bloque y no se abstiene. La siguiente iteración debería explicitar los pares de artículos en la consulta, separar «relación entre artículos» de «respuesta sí/no» y verificar cada cita.
4. Inspeccionar logits reveló un `repetition_penalty` oculto que también afectaba a PC1: el decoding declarado debe comprobarse en la ejecución, no solo en la configuración.
5. Base para la Semana 9: el benchmark versionado, el recuperador comparado, el RAG evaluado y la herramienta con contratos quedan listos para conectarse a un workflow sin reimplementarlos.

---

#### 12. Reproducibilidad y commit evaluado

- **Commit evaluado:** ver [`COMMIT_EVALUADO.md`](COMMIT_EVALUADO.md).
- **Modelos (revisión del Hub):** `intfloat/multilingual-e5-small@614241f`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1@1427fd6`, `Qwen/Qwen2.5-0.5B-Instruct@7ae5576`.
- **Entorno:** Python 3.12.3, macOS 15.4 arm64, CPU. Versiones en `requirements.txt` y en `results/*/run_config.json`.
- **Huellas:** SHA-256 de corpus, consultas, qrels y prompts en `results/exp1/run_config.json` y `results/exp2/run_config.json`.
- **Salidas originales:** `results/exp1/rankings.jsonl`, `results/exp2/generations.jsonl` (con `raw_output`), `results/exp3/traces.jsonl`. Todas las tablas de este README se recalculan desde esos archivos.

#### 13. Fuentes

**Fuentes primarias técnicas**
1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. arXiv:2005.11401. → Arquitectura RAG: memoria no paramétrica, contexto con evidencia y procedencia (sección 8).
2. Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR 3(4). → Parámetros `k1`, `b` y el IDF de BM25 (sección 7).
3. Cormack, G., Clarke, C. & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR. → RRF con `k=60`: fusión por posiciones, porque los scores de BM25 y del coseno no son comparables.
4. Nogueira, R. & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085. → El Cross-Encoder solo sobre los candidatos de la primera etapa (C = B + reranker).
5. Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5). arXiv:2212.03533. → Prefijos `query:`/`passage:` obligatorios y normalización L2.
6. Ainslie, J. et al. (2023). *GQA*. arXiv:2305.13245. → Explica la reducción del KV cache de Qwen (sección 5, S1-S2).

**Fuentes del dominio**
7. Wadden, D. et al. (2020). *Fact or Fiction: Verifying Scientific Claims* (SciFact). EMNLP. arXiv:2004.14974. → Formulación «afirmación + evidencia + veredicto» y abstención cuando no hay evidencia (`insufficient_evidence`).
8. Cohan, A. et al. (2019). *Structural Scaffolds for Citation Intent Classification*. NAACL. arXiv:1904.01608. → El intento de cita (de fondo, de método, comparativo) motiva la tarea de PC1 y la validación de citas.
9. Liu, N. F. et al. (2023). *Lost in the Middle*. arXiv:2307.03172. → Justifica un contexto corto (top-3 chunks) en lugar de entregar más passages.

**Material del curso** (repositorio `kapumota/CC-0F4`): `Semana1/Cuaderno1`, `Lectura1`, `Lectura2`; `Semana2/Cuaderno2`, `Lectura3`; `Semana3/Cuaderno3`; `Semana4/Cuaderno4`, `Laboratorio4`; `Semana5/Cuaderno5`, `Lectura5`; `Semana6/Cuaderno6`, `Lectura6`; `Semana7/Cuaderno7`, `Lectura7`. Las celdas adaptadas se indican en la tabla de la sección 4.

Se usaron herramientas de IA generativa como apoyo para programar y redactar, como permite el curso. Las decisiones, los qrels y la auditoría son responsabilidad del autor.
