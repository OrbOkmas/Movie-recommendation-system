# Semantic Search Pipeline — Usage Guide

Second retrieval approach next to the TF-IDF baseline (`docs/tfidf_pipeline.md`). Same data, same 28 mood queries, same relevance scale, same metrics.

## How it works

| Step | What happens | Why |
|---|---|---|
| Split | `text_embedding` is split at `| Dialogue:` into **metadata + plot** and **dialogue** | the two parts behave very differently (short summary vs. 6,000 words of speech) |
| Chunk | dialogue → windows of 150 words; a tail < 50 words is merged into the previous window | MiniLM reads only ~256 word-pieces; longer text is silently truncated |
| Encode | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalised vectors (float16 on disk) | fast, standard baseline encoder; dot product = cosine similarity |
| Query | the **raw** sentence is encoded (no stop-word removal or lemmatisation) | keeps negation and word order; TF-IDF loses "not" |
| Score | `meta`: cosine with the plot vector; `subtitles`: aggregate chunk cosines (`max`, `topk_mean`, `mean`); `combined`: `w·meta + (1−w)·subtitles` | lets us test whether subtitles add anything over the plot |

All variants use **one index**, they differ only at search time.

| Variant (runner name) | Source | Aggregation |
|---|---|---|
| `sem_meta` | metadata + plot | – |
| `sem_subs_max` | subtitles | best chunk |
| `sem_subs_top3` | subtitles | mean of 3 best chunks |
| `sem_combined` | 0.5 · meta + 0.5 · subtitles | mean of 3 best chunks |

Each subtitle-based result contains `best_chunk`: the dialogue passage that matched the query best. This is the explanation of a recommendation and the main tool for error analysis.

## Commands (repository root)

```bash
pip install sentence-transformers pyarrow

# 1) build the index (sample: ~1–3 min on CPU, seconds on GPU)
python -m src.semantic train --data data/processed/sample_clean_movies.csv --output models/semantic_sample

# 2) single query
python -m src.semantic search --index models/semantic_sample --query "dark but not scary" --source combined

# 3) all 28 queries x 4 variants; adds new pairs to evaluation/mood_judgments.csv
python -m src.run_semantic_eval run --index models/semantic_sample

# 4) after filling `relevance` (0/1/2): TF-IDF vs. semantic variants
python -m src.run_semantic_eval metrics
```

Full data: use `notebooks/03_semantic_search_colab.ipynb` on a Colab T4 GPU (~2 million chunks).

## Outputs (`evaluation/results/`)

| File | Content |
|---|---|
| `sem_<variant>_mood_rankings.csv` | top-5 per query with `score`, `meta_score`, `subtitle_score`, `best_chunk` |
| `semantic_query_status.csv` | ok / no_matches / error per query and variant |
| `method_overlap.csv` | share of each top-5 list shared with TF-IDF (available before human labels) |
| `comparison_per_query.csv`, `comparison_metrics.csv` | P@5, MRR@5, nDCG@5 (judged pool), only on queries fully judged for **all** methods |

Metric definitions are identical to `notebooks/02_mood_query_evaluation.ipynb`.

## Known limitations

- Fixed word windows ignore scene boundaries; one chunk may mix two scenes.
- MiniLM was trained on general sentence pairs, not on mood/atmosphere descriptions or movie dialogue. Dialogue talks *within* the mood ("Get out of the house!"), whereas queries talk *about* the mood ("eerie, unsettling").
- Dense retrieval always returns 5 movies, even for meaningless queries; there is no "no match" signal without a threshold.
- `--model debug-hashing` exists only to test the code offline; never report its results.
- Do not commit `chunk_text.parquet` or the `.npy` files of the full index (size and subtitle copyright).
