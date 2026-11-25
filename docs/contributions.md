# Team Contributions

**Project:** Movie Recommendation System

| Member | Main responsibility | Actual contributions |
|---|---|---|
| Member 1 — [Name] | Project coordination and integration | [Completed tasks] |
| Member 2 — [Name] | Data collection and preprocessing | [Completed tasks] |
| Member 3 — [Name] | Baseline recommendation method | [Completed tasks] |
| Member 4 — Tomáš | Semantic recommendation method (Semantic Search 1) | Designed and implemented sentence-embedding search with subtitle chunking (`src/semantic.py`): plot vs subtitle representations, chunk-score aggregation, explanations via matched dialogue; evaluation runner and notebook with the same metrics as the baseline (`src/run_semantic_eval.py`, `notebooks/04_semantic_mood_evaluation.ipynb`); pooled blind labelling sheet; main comparison with TF-IDF fixed before labelling, ablation of four variants, significance tests; Colab notebook for the full-data run; documentation (`docs/semantic_pipeline.md`) and failure-journal experiments 3–7.|
| Member 5 — [Name] | Evaluation and error analysis | [Completed tasks] |
| Member 6 — [Name] | Documentation, testing and application | [Completed tasks] |

## Shared Work

**Data and methodology:** [Who contributed?]

**Comparison and error analysis:** [Who contributed?]

**Limits and defense:** [Who contributed?]

**Failure journal:** [Who maintained the records?]

**Integration and final presentation:** [Who contributed?]

All members are expected to understand and justify the project's important methodological decisions.
## Matyáš — Semantic Search v2 (improved semantic retrieval) and its evaluation

**Role.** I designed, implemented and evaluated the second semantic retrieval model (`src/semantic_v2/`), whose job is to beat semantic search v1 and the TF-IDF baseline on the same queries and judgments. Every part below is mapped to the grading pillar it serves.

| Pillar | My work | Where to see it |
|---|---|---|
| **Data & Methodology** | Profile of the full corpus (51,616 movies, 2.25 M chunks) and the design decisions derived from it: separate field vectors, explicit handling of missing fields, chunking, hub correction with a background query bank, z-score fusion, query processing (negation, clauses, spelling, statuses). Streaming, sharded and resumable index build for the 2.4 GB corpus. Encoder choice based on measured encoding time. | `docs/semantic_v2_plan.md` (decisions D1–D10 with rejected alternatives), `docs/semantic_v2.md`, `evaluation/results/v2/data_profile.json`, `encoder_timing.csv` |
| **Comparison & Error analysis** | Evaluation protocol: separate dev set of 50 queries written for tuning (33 core, 17 robustness) and the team's M01–M28 as test; corrected nDCG (ideal from the judged pool) next to the team metric, bootstrap intervals, paired differences; label-free diagnostics (hub concentration, typo/paraphrase stability, status accuracy, latency); generated results and error-analysis report on concrete failing queries. Comparison of v2 with TF-IDF and the four v1 variants on identical judgments. | `docs/semantic_v2_results.md`, `scripts/make_results_report.py`, `src/semantic_v2/evaluate.py`, `evaluation/dev_queries.json` |
| **Process reflection & Failure journal** | Experiments 7–13: damaged data copy, false negations, wrong spelling corrections, clause over-splitting, abstention by score failing, encoder chosen by compute, flaw of the shared nDCG. Each with evidence, change and lesson. | `docs/failure-journal.md` |
| **Critical thinking & Defense** | Limitations written down (rule-based negation, real-word typos, residual hubs, untuned weights, small sample, design informed by test queries); conservative defaults (the parser leaves unsure cases alone; the whole query is always scored); 70 automated tests that fix the expected behaviour. | `docs/semantic_v2.md` ("Known limitations"), `tests/` |

**Interfaces to teammates.** The app can call `SemanticSearcher.search()` directly and receives a Python list of results (no CSV needed; see `docs/semantic_v2.md`, "Using it from the app"). I reused the team's queries, relevance scale and judgment sheet format so that results are comparable with Tomáš's v1 evaluation.