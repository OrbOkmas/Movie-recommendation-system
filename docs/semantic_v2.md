# Semantic Search v2 — Usage Guide and Documentation

Improved semantic retrieval next to the TF-IDF baseline (`docs/tfidf_pipeline.md`) and semantic search v1 (`docs/semantic_pipeline.md`). Same data, same 28 mood queries, same relevance scale (0/1/2). Author: Matyáš. Design decisions and alternatives: `docs/semantic_v2_plan.md`. Results on the 100-movie sample (generated from the files, not typed in): `docs/semantic_v2_results.md`. What failed on the way: `docs/failure-journal.md`, experiments 7–13.

## What changed compared with v1

| v1 problem (from `docs/failure-journal.md`, exp. 6) | v2 answer |
|---|---|
| one string "title + genres + keywords + plot" per movie; short plots and title words dominate | **separate vectors** for plot, keywords and genres; the title is not embedded. A missing field has no vector and no score (no empty-string embedding) |
| raw cosines of different sources mixed 0.5/0.5 | each source is **z-scored per query** and combined with weights; weights are renormalised over the sources a movie really has |
| one movie in 19 of 27 top-5 lists (hub) | **hub correction**: each movie's average similarity to 800 generic background queries is subtracted (subtitle channel) |
| negation ("no horror") is ignored by the encoder | **negation parser** + penalty `λ·max(0, z_neg − τ)`; optional genre/keyword based negation (`--meta-negation`) |
| multi-aspect queries ("cozy and a little grim") | whole query **plus soft-minimum over clauses**; splitting is deliberately conservative |
| misspellings return unrelated films | **spell correction** (SymSpell + corpus vocabulary), repeated letters, shouting, punctuation |
| always returns 5 films, even for "Surprise me" or gibberish | explicit **statuses** and rule-based abstention for vague / unrecognised queries |
| MiniLM-L6 for everything | `bge-base` for the short fields, `bge-small` for the 2.25 M subtitle chunks (measured compute budget, see failure journal exp. 12) |

## How it works

| Step | What happens | Why |
|---|---|---|
| Load | the cleaned table is streamed (parquet/CSV); a movie with no plot, no keywords and no usable dialogue is skipped | RAM-safe for the 2.4 GB parquet; nothing to match on |
| Fields | `plot` (≤250 words), `keywords` ("Keywords: a, b, c", ≤15), `genres` ("Genres: Drama, Comedy"), each encoded separately | one concatenated string hid which part matched |
| Chunks | dialogue → 150-word windows (tail < 50 words merged); one vector per chunk, per-movie score = mean of the 3 best chunks | one vector per movie cannot hold 6,000 words; 3 best chunks are more robust than 1 |
| Encode | `bge-base-en-v1.5` (fields, 768-d) and `bge-small-en-v1.5` (chunks, 384-d); L2-normalised, float16 on disk, sharded | compute budget |
| Hub offsets | 800 background queries (templated mood phrases + random plot sentences, never evaluation queries); offset = mean of the 10 highest scores of a movie | removes movies that are similar to everything |
| Query | normalise → length check → spell correction → negation parse → clause split → encode | see query processing |
| Score | `z_f = (s_f − mean_f(query)) / std_f(query)`; `z(m) = Σ w_f z_f / Σ w_f`; `score = α·z_whole + (1−α)·softmin(z_clauses) − λ·max(0, z_neg − τ)` | formulas in `docs/semantic_v2_plan.md`, section 5 |

## Data (profile of the full corpus, `evaluation/results/v2/data_profile.json`)

51,616 movies (all rankable), 2,254,879 chunks. Missing: plot 0.5 %, keywords 27.6 %, genres 1.3 %, dialogue 0.2 %. Plot: median 43 words. Years 1895–2026 (median 2003). Genres: Drama 50 %, Comedy 29 %, Romance 17 %, Thriller 17 %, Action 15 %, Horror 11 %.

## Commands (repository root)

```powershell
pip install -r requirements-semantic-v2.txt        # CUDA build of torch first for the GPU
python -m spacy download en_core_web_sm
python -m pytest -q                                 # 70 offline tests, no model download

python -m src.semantic_v2 profile --data data/processed/clean_movies.parquet --out evaluation/results/v2/data_profile.json
python -m src.semantic_v2 train --data data/processed/clean_movies.parquet --output models/v2_bgebase_bgesmall `
    --field-model bge-base --chunk-model bge-small --exclude-queries evaluation/dev_queries.json evaluation/mood_queries.json
python -m src.semantic_v2 search --index models/v2_bgebase_bgesmall --query "something scary but without any gore" --config configs/v2_default.json
python -m src.semantic_v2 run --index models/v2_bgebase_bgesmall --queries evaluation/dev_queries.json evaluation/mood_queries.json --name v2_A_full --top-k 10 --config configs/v2_default.json
python -m src.semantic_v2 diag --rankings evaluation/results/v2/v2_A_full_rankings.csv --queries evaluation/dev_queries.json evaluation/mood_queries.json --status evaluation/results/v2/v2_A_full_query_status.csv
python -m src.semantic_v2 agreement --a evaluation/mood_judgments_v2.csv --b evaluation/judgments_<initials>.csv
```

Sample run and evaluation in one go: `scripts/run_sample_eval.ps1` (builds or reuses `models/v2_sample_bge`, runs the 28 queries, computes metrics for TF-IDF, semantic v1 and v2 on the same judgments, writes `docs/semantic_v2_results.md`). Full-corpus builds of the three encoder variants: `scripts/build_full_indexes.ps1`.

`train` is resumable (shards of 2,000 movies; rerun the same command after an interruption). Changing models, chunk size or `--max-chunks-per-movie` needs a new index; every other setting is a search-time option (`--config`, `--w-plot`, `--lam`, `--no-clauses`, `--meta-negation`, …).

## Query processing and statuses

`search()` never raises on user input and returns `status`, `warnings`, `message`, `results`.

| Status | When | Results |
|---|---|---|
| `ok` | normal query | yes |
| `ok_with_warning` | query longer than 60 words, or a spelling correction was applied (the warning says which) | yes |
| `empty` | empty / whitespace / not text | no |
| `negation_only` | only says what is *not* wanted ("Nothing too scary and nothing too sad") | no |
| `low_confidence` | too vague ("Surprise me", "A good movie", "I feel like watching something": warning `too_vague`), mostly unrecognised words (`unrecognized_words`), or no letters at all | no (unless `return_all=True`, which `run` uses for evaluation) |

Queries above 150 words are truncated (warning `query_truncated`). English only.

## Settings (`configs/v2_default.json`, `SearchConfig`)

| Setting | Default | Meaning |
|---|---|---|
| `weights` | plot 1.0, keywords 0.4, genres 0.2, subtitles 0.5 | source weights (**placeholders, not tuned yet**) |
| `hub_fields`, `hub_gamma` | subtitles, 1.0 | which sources get the hub correction, and how strongly |
| `alpha`, `softmin_T` | 0.5, 1.0 | whole query vs. soft-min over clauses |
| `lam`, `tau` | 1.0, 0.5 | negation penalty strength and tolerated z-score of the negated term |
| `meta_negation`, `meta_neg_penalty` | false, 2.0 | genre/keyword based negation (ablation) |
| `use_clauses` | true | clause splitting on/off |
| `min_confidence` | 0 (off) | numeric abstention threshold; **does not work with BGE** (failure journal, exp. 11) |

## Using it from the app

```python
from src.semantic_v2.config import SearchConfig
from src.semantic_v2.search import SemanticSearcher

searcher = SemanticSearcher.from_dir("models/v2_bgebase_bgesmall", SearchConfig.from_json("configs/v2_default.json"))  # once
r = searcher.search("something cozy and a little bit grim", top_k=10, snippets=False)
r.status, r.message, r.warnings      # show to the user
r.results                            # list of dicts: rank, movie_id (IMDb id), title, year, genres, score,
                                     # plot_score, keywords_score, genres_score, subtitle_score, neg_penalty
```

The app can take `r.results[:3]` directly; no CSV is needed. The index is several GB and is not in git (share it via the team drive); the `chunk_text.parquet` files are only needed for dialogue snippets (`snippets=True`). After the first run, `HF_HUB_OFFLINE=1` avoids network calls at start-up. Measured on the full index (RTX 2060 machine, GPU for the query encoder): median 4.5 s per query, p95 5.5 s, dominated by scoring 2.25 M chunks.

## Outputs (`evaluation/results/v2/`)

| File | Content |
|---|---|
| `data_profile.json` | profile of the full corpus |
| `encoder_timing.csv` | measured encoding time per encoder (500 movies) |
| `v2_A_full_*` | 78 queries (50 dev + 28 test) on the full index: rankings (top-10), status and latency, diagnostics |
| `v2_A_sample_*`, `sample_*` | the 28 queries on the 100-movie sample, metrics and summaries |
| `docs/semantic_v2_results.md` | generated report: metrics with bootstrap intervals, paired differences, by category, error analysis |

## Evaluation protocol

- **Test set:** M01–M28 (the team's queries). **Dev set:** `evaluation/dev_queries.json`, 50 queries written separately (33 core, 17 robustness), with gold annotations that double as unit tests.
- **Metrics:** P@5, MRR@5, nDCG@5 as in the team notebook, plus an nDCG@5 whose ideal ranking comes from all judged movies of the query (the team version takes the ideal from the retrieved list itself, so it cannot see relevant movies that were missed), with per-query bootstrap 95 % intervals. Label-free: hub concentration, typo/paraphrase stability, status accuracy, latency.
- **Labels:** pooled judgments of all methods. Vague queries M25 and M27 are scored 1 for every film and reported separately.

## Known limitations

- Dialogue talks *within* a mood, queries talk *about* it; the subtitle channel adds little for mood queries (see the by-label table in the results).
- Negation parsing is rule based: "anything but…", sarcasm and double negation are not handled. A false negation could remove a good film, so the parser is conservative.
- Spelling: real-word typos ("griping") and names ("grimm") are not corrected; "mistree" becomes "mistress".
- Residual hubs: obscure movies with generic titles and very short plots still appear for many queries (up to 6 of 75 queries on the full index).
- Weights, λ and τ are untuned placeholders; the encoder split (bge-base / bge-small) is a decision under a compute budget, not a measured optimum.
- The sample evaluation has 100 movies and about 25 judged queries; most pooled candidates are poor matches, so differences are within noise unless the intervals say otherwise.
- The corpus is OpenSubtitles + TMDB: English only, drama-heavy, many obscure titles.
- v2 was designed after seeing v1's failures on M01–M28, so the sample result is not an untouched test.

## Not done yet

Tuning of weights, λ and τ on the dev set (needs dev labels), the full ablation ladder (fusion, hub, negation, clauses, encoder), and the cross-encoder reranker (backup plan D10 in the plan, only if needed).
