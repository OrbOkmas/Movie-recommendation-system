# Semantic search v2 - how to run it

Design and decisions: `docs/semantic_v2_plan.md`. All commands from the repository root; `src/semantic_v2/` is a
self-contained package (it does not import v1). Set up once:

```powershell
pip install -r requirements-semantic-v2.txt        # install the CUDA build of torch first for the GPU
python -m spacy download en_core_web_sm
python -m pytest -q                                 # 53 offline tests, no model download needed
```

## Workflow
```powershell
# 0. profile the real data (replaces guesses about missing rates / lengths)
python -m src.semantic_v2 profile --data data/processed/clean_movies.parquet --out evaluation/results/v2/data_profile.json

# 1. build an index (resumable: re-run the same command after a crash; --overwrite to start over)
python -m src.semantic_v2 train --data data/processed/sample_clean_movies.csv --output models/v2_sample `
    --field-model minilm-l6 --chunk-model minilm-l6 --exclude-queries evaluation/dev_queries.json evaluation/mood_queries.json

# 2. try a query
python -m src.semantic_v2 search --index models/v2_sample --query "something scary but without any gore"

# 3. run dev + test queries, write rankings and a v2-specific judgment sheet (never the team's file)
python -m src.semantic_v2 run --index models/v2_sample --queries evaluation/dev_queries.json evaluation/mood_queries.json `
    --name v2_baseline --judgments evaluation/mood_judgments_v2.csv

# 4. label-free diagnostics, labeled metrics (after humans fill `relevance`), abstention threshold
python -m src.semantic_v2 diag --rankings evaluation/results/v2/v2_baseline_rankings.csv --queries evaluation/dev_queries.json --status evaluation/results/v2/v2_baseline_query_status.csv
python -m src.semantic_v2 metrics --rankings v2_baseline=evaluation/results/v2/v2_baseline_rankings.csv --judgments evaluation/mood_judgments_v2.csv
python -m src.semantic_v2 calibrate --index models/v2_sample --queries evaluation/dev_queries.json
```
Ablations are flags of `search`/`run`/`calibrate`: `--w-plot/--w-keywords/--w-genres/--w-subtitles`, `--hub-fields`,
`--hub-gamma`, `--alpha`, `--lam`, `--tau`, `--no-clauses`, `--meta-negation`, `--min-confidence`, `--no-spell`.
Changing models, chunk size or `--max-chunks-per-movie` needs a new index; everything else does not.

## Status
Implemented: field-wise index with missing-field handling, sharded resumable encoding, hub correction, z-score fusion,
negation penalty (+ optional metadata negation), clause soft-min, query processor (spelling, long-query warning,
negation, clauses, statuses), abstention + calibration, evaluation (team metrics, pool nDCG, bootstrap CI, label-free
diagnostics), data profile.
Not yet: weight/lambda tuning script (needs dev labels), real-model runs (encoder comparison), cross-encoder rerank (backup plan D10).
Not verified here: anything with real sentence-transformer models or a GPU (all 53 tests use the offline hashing encoder).
