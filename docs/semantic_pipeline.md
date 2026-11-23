# Semantic Search 1 — Sentence Embeddings with Subtitle Chunking

Author: Tomáš · Code: `src/semantic.py`, `src/run_semantic_eval.py` · Evaluation: `notebooks/04_semantic_mood_evaluation.ipynb`

The second retrieval approach next to the TF-IDF baseline (`docs/tfidf_pipeline.md`). Same 100-movie sample, same 28 mood queries, same 0/1/2 relevance scale and the same metric code, so the results are directly comparable.

---

## 1. Model and why we chose it

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — a pre-trained sentence-embedding model (6 transformer layers, 384-dimensional vectors, ~90 MB). It maps a sentence to a vector so that sentences with a similar *meaning* get similar vectors, even when they share no words.

**Why:**
- TF-IDF only matches words. Mood queries are paraphrastic ("gloomy", "dark", "bleak" mean similar things) and often share no vocabulary with plots or dialogue. Embeddings were the natural next step to test whether *meaning* retrieves better than *words*.
- MiniLM is the standard fast baseline encoder: it runs on a CPU, needs no training on our side, and is small enough for ~2 million subtitle chunks on a Colab GPU.
- The query is encoded **as written** (no stop-word removal or lemmatisation), so words like "not" reach the model — TF-IDF removes them. Whether the model actually uses them was one of the questions we tested.

## 2. How it works

| Step | What happens | Why |
|---|---|---|
| Split | The prepared `text_embedding` column is split at `\| Dialogue:` into **metadata + plot** (title, genres, keywords, plot) and **dialogue** (subtitles). | A 70-word summary and 6,000 words of speech behave very differently. |
| Chunk | Dialogue is cut into windows of **150 words**; a tail shorter than 50 words is merged into the previous window. Sample: 4,155 chunks, 41.6 per movie. | MiniLM reads at most 256 word-pieces (~150–200 words) and silently truncates the rest. One vector per movie would see < 2 % of the dialogue. |
| Encode | One vector for metadata + plot, one vector per chunk; all L2-normalised, stored as float16. | Dot product = cosine similarity; float16 halves disk and RAM. |
| Query | The raw sentence is encoded with the same model. | Keeps negation and word order. |
| Score | `meta`: cosine(query, plot vector). `subtitles`: aggregate the chunk cosines of a movie (`max` = best chunk, `topk_mean` = mean of 3 best). `combined`: 0.5 · meta + 0.5 · subtitles(top 3). | Lets us test whether subtitles add information beyond the plot. |
| Explain | Each subtitle-based result carries `best_chunk` — the dialogue passage that matched the query best. | Makes every recommendation inspectable; the main tool of the error analysis. |

All four variants share **one index**; they differ only in how scores are computed at search time, so a new scoring idea never requires re-encoding the data.

| Variant | Source | Role in the evaluation |
|---|---|---|
| **`sem_combined`** | 0.5 · plot + 0.5 · subtitles (top 3) | **Main model.** Compared with TF-IDF (both use plot + subtitles). Fixed on 20 Nov, before any semantic labels existed. |
| `sem_meta` | plot + metadata only | ablation |
| `sem_subs_max` | subtitles, best chunk | ablation |
| `sem_subs_top3` | subtitles, mean of 3 best chunks | ablation |

## 3. Results (100-movie sample, human judgments)

**Main comparison** (23 queries for which both methods returned five judged movies):

| Method | Precision@5 | MRR@5 | nDCG@5 (judged pool) |
|---|---:|---:|---:|
| TF-IDF | 0.322 | 0.456 | 0.481 |
| **Semantic Search 1 (`sem_combined`)** | **0.461** | **0.638** | **0.631** |

- Semantic search is better on 14 queries, TF-IDF on 6, 3 ties (nDCG@5).
- Difference in Precision@5: +0.14, bootstrap 95 % CI [0.03, 0.25], Wilcoxon signed-rank p = 0.045.
- Difference in nDCG@5: +0.15, bootstrap 95 % CI [0.03, 0.28], Wilcoxon p = 0.07.
- With 23 queries this is a consistent but not definitive improvement.

**Ablation — does the text source matter?** (all 27 non-empty queries)

| Variant | Precision@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| `sem_combined` (plot + subtitles) | **0.467** | 0.630 | 0.641 |
| `sem_meta` (plot only) | 0.422 | **0.631** | **0.657** |
| `sem_subs_top3` (subtitles only) | 0.422 | 0.548 | 0.590 |
| `sem_subs_max` (subtitles only) | 0.407 | 0.504 | 0.587 |

Subtitles alone are weaker than the plot alone; adding them to the plot helps only Precision@5 slightly. The project hypothesis that dialogue carries the mood of a film is only partly supported.

**By query type (Precision@5, same 23 queries):** the gain is largest for direct moods (0.70 vs 0.30, 4 queries) and mood themes (0.43 vs 0.23), small for subtle "vibe" queries (0.37 vs 0.33) and negation (0.47 vs 0.37). For the vague edge case "A genre I cannot name, just something different" (M27) TF-IDF is better (0.60 vs 0.20).

**Successful example — M09 "A warm romance about two people finding each other unexpectedly."**
Semantic: *Perfect Match* (grade 2; best chunk is a love-song lyric), *Extreme Dating* (1), … TF-IDF: *Dangerous Love Affairs*, *Bhoot* (horror), *Justine and the Whip*, *Born to Kill*, *Salt and Fire* — all graded 0.

**Failure examples**
- **Hub movie:** *Ways to Live Forever* (a drama about a boy with leukaemia) is in the top 5 of 14 of 27 queries and is graded 0 in 10 of them — e.g. rank 1 for "a cozy, comforting movie for a rainy evening" (M02). Its dialogue *talks about* horror films, fear, death and feelings, so some chunk is close to almost any mood query ("…there's nothing terrifying about it. Look, you wanted horror films…").
- **Negation:** "No horror, just tension and psychological mystery" (M23) — all five semantic results (and all five TF-IDF results) are horror films or irrelevant: 0 relevant. The encoder places "no horror" close to "horror".
- **Data noise:** for M23 the best chunk of *Scared* is subtitle credits ("German subtitles translated by …"). Preprocessing did not remove translator credits.

## 4. Problems and how we solved them

| Problem | Solution | Journal |
|---|---|---|
| One vector per movie would see < 2 % of the dialogue (256-token limit). | Separate plot vector + 150-word dialogue chunks with score aggregation. | Exp. 3 |
| Tiny trailing chunks could win under max-aggregation; first fix had an off-by-one bug. | Tail < 50 words merged into the previous chunk; covered by a unit test; aggregation verified against a pandas reference. | Exp. 4 |
| No model download / GPU in the development environment; ~21 h estimated on CPU for the full data. | Pipeline tested offline with a stand-in encoder; real runs on a laptop (sample, 152 s) and Colab GPU (full data). | Exp. 5 |
| Comparing four semantic variants with one TF-IDF favours "semantic" if the best variant is picked afterwards. | Main comparison (`sem_combined` vs TF-IDF) fixed before labelling; other variants reported as ablation; all variants reported. | Exp. 7 |
| Judging the same movie twice / judge knowing the method. | Pooled, blind labelling sheet; pairs already judged for TF-IDF are reused, not re-judged. | Exp. 7 |
| Misspelled queries. | Spelling correction tested (3 variants); not adopted yet — only one test query is affected and a hand-made lexicon risks tuning to the test set. | Exp. 8 |
| Hub movies, negation. | **Not solved** — documented as limitations; hub correction (score normalisation per movie) is the next step. | Exp. 6–7 |

## 5. Limitations

- **Single judge, two sessions:** all labels come from one team member, which keeps the scale consistent but subjective. The semantic sheet was labelled later and contained only semantic results, so the judge could tell which method they came from (blinding only partial). A second judge on a sample of pairs would measure agreement.
- **Judged-pool nDCG:** the ideal ranking is built only from judged retrieved movies (same definition as for TF-IDF).
- **Small, non-random sample** (100 movies) and 23–27 evaluated queries; "An adventurous story full of wonder…" (M11) has no relevant movie for any method and "the vibe of an old, forgotten seaside town" (M15) only one across all methods — possibly no such movie exists in the sample.
- **Underspecified query M25** ("I feel like watching something"): semantic search returns five movies and the judge graded all of them 1; TF-IDF returned nothing. M25 is excluded from the main comparison.
- Fixed word windows ignore scene boundaries; MiniLM was not trained on film dialogue or mood descriptions.

## 6. How to run

```bash
pip install sentence-transformers pyarrow matplotlib scipy

# build the index (sample: ~2.5 min on a laptop CPU)
python -m src.semantic train --data data/processed/sample_clean_movies.csv --output models/semantic_sample

# one query
python -m src.semantic search --index models/semantic_sample --query "dark but not scary" --source combined --top-k 10
```

Evaluation: open `notebooks/04_semantic_mood_evaluation.ipynb` and **Run All**. It runs the 28 queries, writes `evaluation/results/sem_*_mood_rankings.csv`, creates/extends the labelling sheet `evaluation/mood_judgment_semantic.csv` (only pairs without a grade) and computes the metrics from `evaluation/mood_judgment_tfidf.csv` + `evaluation/mood_judgment_semantic.csv`. Scripted equivalent: `python -m src.run_semantic_eval run --index models/semantic_sample` and `python -m src.run_semantic_eval metrics`.

**Using the model from Python (e.g. in the application):**

```python
from src.semantic import SemanticMovieRecommender
model = SemanticMovieRecommender.load("models/semantic_sample")      # load once
results = model.search("something dark but not scary", top_k=10, source="combined")
# -> list of dicts: rank, movie_id, title, score, meta_score, subtitle_score, genres, release_year, best_chunk
```

Full data on a GPU: `notebooks/03_semantic_search_colab.ipynb`.

## 7. Output files (`evaluation/results/`)

| File | Content |
|---|---|
| `sem_<variant>_mood_rankings.csv` | top 5 per query with `score`, `meta_score`, `subtitle_score`, `best_chunk` |
| `semantic_query_status.csv` | ok / no_matches / error per query and variant |
| `comparison_per_query.csv`, `comparison_metrics.csv` | P@5, MRR@5, nDCG@5 per query and averaged |
| `semantic_failure_candidates.csv` | queries with no relevant movie for `sem_combined`, with matched chunks |
