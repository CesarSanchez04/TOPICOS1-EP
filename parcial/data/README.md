### Datos del examen parcial (`corpus_version = parcial-v1`)

#### Archivos

| Archivo | Contenido |
|---|---|
| `raw/arxiv_raw.json` | Respuesta de la API de arXiv tal como se descargó (no se edita a mano) |
| `corpus.jsonl` | 36 documentos y 135 passages (formato de `Semana4/datos/corpus_semana4.jsonl`) |
| `queries.jsonl` | 22 consultas estructuradas (20 para recuperación y 2 `no_answer` para abstención) |
| `qrels.json` | 56 juicios graduados sobre `passage_id` (2 = evidencia directa, 1 = parcial) |
| `reference_answers.jsonl` | Relación gold y hechos clave de las 12 preguntas del experimento 2 |

#### Procedencia y permisos

- **Fuente:** API pública de arXiv (`export.arxiv.org/api/query`), descargada el 2026-10-27.
- **Contenido:** título, autores, fecha de la primera versión, versión vigente (`arxiv_version`), categoría y abstract.
- **Licencia:** arXiv distribuye los metadatos, incluidos los abstracts, bajo **CC0 1.0** ([términos de la API](https://info.arxiv.org/help/api/tou.html)). No se descargaron PDFs ni texto completo.
- **Identificadores estables:** `doc_id` = arXiv ID (por ejemplo, `2203.15556`); `passage_id` = `<arXiv ID>-P<nn>`. La versión exacta de cada artículo queda en `arxiv_version` y la huella SHA-256 de cada archivo, en `results/exp1/run_config.json`.

#### Selección

Se eligieron, **antes de escribir las consultas**, 12 debates de ML/NLP en los que dos o más artículos coinciden, discrepan o se complementan sobre una misma afirmación. Es el mismo dominio de PC1 (Kaplan vs. Chinchilla, RAG vs. fine-tuning, MHA/MQA/GQA, CoT, DPO vs. PPO, cuantización, RoPE, atención eficiente), ampliado con recuperación densa vs. BM25, in-context learning y LoRA. La lista exacta, con el tema de cada artículo, está en `raw/arxiv_raw.json`.

#### Segmentación

El cuaderno (§3) divide cada abstract en oraciones (sin cortar «e.g.», «i.e.» ni «et al.») y las agrupa hasta llegar a 35 palabras o más. Un resto de menos de 17 palabras se une al passage anterior. El corpus no se infla: `validate_benchmark` rechaza passages o textos duplicados.

#### Protocolo de anotación

1. Se leyeron los 135 passages completos.
2. Se escribieron 20 consultas con dificultad declarada: 10 `clear`, 3 `ambiguous`, 5 `hard` y 2 `distractor`. Las difíciles requieren evidencia de dos artículos que se contradicen; las de distractor tienen passages léxicamente cercanos pero no relevantes (por ejemplo, «emergent features» de LLM.int8 frente a «emergent abilities»).
3. Se anotaron los qrels **antes de ejecutar cualquier sistema** y no se modificaron después. Grado 2: el passage responde directamente; grado 1: aporta contexto o la mitad del argumento.
4. Se agregaron 2 consultas `no_answer` (Mamba, Llama 3) cuya respuesta no está en el corpus. No tienen qrels, se excluyen de Recall/MRR/nDCG y sirven para evaluar la abstención en el experimento 2.

La evidencia gold no se generó con el sistema evaluado.

#### Limitaciones

- Solo abstracts: algunos hallazgos (por ejemplo, la velocidad de GPTQ) aparecen en un único passage.
- Un solo anotador y sin medición de acuerdo entre anotadores.
- Las consultas están en inglés porque el corpus lo está; con consultas en español, BM25 quedaría artificialmente penalizado.
- 20 consultas permiten una evaluación docente, **no prueban generalización estadística**.

#### Regla experimental

No modificar `corpus.jsonl`, `queries.jsonl` ni `qrels.json` después de observar resultados. Cualquier cambio constituye una nueva versión del benchmark.
