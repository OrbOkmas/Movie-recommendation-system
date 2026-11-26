# Movie Recommendation System

## Project Overview

This project is developed as part of the Text Analytics 2 course at Prague University of Economics and Business.

Our goal is to develop and evaluate a movie recommendation system using natural language processing techniques. The user describes a **mood** in free text ("something gloomy and thrilling", "no horror, just tension") and the system returns five matching movies. We test whether ranking by meaning beats ranking by shared words, and whether subtitle dialogue adds mood information beyond the plot description.

## Team Members

| Member | Responsibility |
|---|---|
| Vojta | Data collection, TMDB alignment and preprocessing |
| Eliška | TF-IDF baseline, shared test queries, evaluation notebook, repository setup |
| Tomáš | Semantic Search v1 (sentence embeddings) |
| Matyáš | Semantic Search v2 and its evaluation |
| Sam | Relevance labelling, coordination and integration of the prefinal package |
| Matouš | Streamlit application |

Half-page summary: `docs/contributions.md`; per-person details: `docs/contributions_detailed.md`.

**Prefinal package:** [`docs/report.md`](docs/report.md) (main report), [`docs/failure-journal.md`](docs/failure-journal.md), [`docs/contributions.md`](docs/contributions.md). The state handed in for peer review is the git tag `prefinal`.

## Project Objectives

- Prepare and explore a movie dataset.
- Implement at least two recommendation approaches.
- Compare their performance using suitable evaluation metrics.
- Analyse successful recommendations and failure cases.
- Discuss practical limitations and improvements.

## Dataset

**Source:** English subtitles from [OPUS OpenSubtitles v2018](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles) (downloaded by `src/extract_opus.py`), aligned by IMDb id with metadata from [TMDB](https://www.themoviedb.org/) (`src/aligment.py`, `src/enrich_tmdb_details.py`; needs a TMDB API token in `.env`).

**Description:** 51,616 movies (1895–2026, median 2003). Per movie: title, year, genres, keywords, plot and cleaned subtitle dialogue (median 6,205 words). Missing values: plot 0.5%, keywords 27.6%, genres 1.3%, dialogue 0.2%. The corpus is drama-heavy (Drama 50%, Comedy 29%, Horror 11%). All labelled experiments use a 100-movie sample (`data/processed/sample_clean_movies.csv`, the first 100 movies with valid subtitles, not a random sample).

**Data preparation:** `src/preprocessing.py` removes subtitle noise, cleans metadata and builds two text fields: `text_tfidf` (lowercased, spaCy tokens, stop words removed, WordNet lemmas) for the baseline, and `text_embedding` (title, genres, keywords, plot and dialogue as written) for semantic search. Known noise: translator credits and song lyrics remain inside the subtitles.

## Methodology

### Approach 1: TF-IDF Baseline and Mood Evaluation

The lexical movie recommendation baseline is implemented in `src/baseline.py`, with consistent query preprocessing in `src/tfidf_text.py`.

The notebook `notebooks/02_mood_query_evaluation.ipynb` runs the shared mood queries, exports ranked recommendations, and supports human relevance judgments and evaluation metrics.

For setup instructions, training and search commands, evaluation procedures, full-dataset execution, and troubleshooting, see the [TF-IDF Pipeline Usage Guide](docs/tfidf_pipeline.md).

### Approach 2: Semantic Search 1 (sentence embeddings)

Implemented in `src/semantic.py`, evaluated in `notebooks/04_semantic_mood_evaluation.ipynb` (`src/run_semantic_eval.py` is the scripted version).

Movies and queries are encoded with the pre-trained model `all-MiniLM-L6-v2`, so a query can match a movie by meaning without sharing words. The model reads only about 200 words, so each movie gets one vector for metadata and plot and one vector per 150-word subtitle chunk (about 42 per movie). Final score = 0.5 × plot similarity + 0.5 × mean of the 3 best subtitle chunks. Unlike TF-IDF, the query is not stop-word filtered or lemmatised, and each result comes with the dialogue passage that matched. See [Semantic Search 1 documentation](docs/semantic_pipeline.md).

**Evaluation (both approaches):** the same 28 mood queries (`evaluation/mood_queries.json`; direct moods, mood themes, subtle vibes, negation, edge cases), one judge grading each query/movie pair 0 / 1 / 2 in pooled sheets (`evaluation/mood_judgment_tfidf.csv`, `evaluation/mood_judgment_semantic.csv`), and Precision@5, MRR@5 and nDCG@5.

### Approach 3: Semantic Search v2

Implemented in `src/semantic_v2/` (70 tests in `tests/`). Improved semantic retrieval: separate plot/keyword/genre vectors, calibrated fusion, hub correction, negation and typo handling, rule-based abstention. Usage and design: `docs/semantic_v2.md`, `docs/semantic_v2_plan.md`; results: `docs/semantic_v2_results.md`; run everything on the sample with `scripts/run_sample_eval.ps1`.

## Results

On the 23 queries where both methods returned five judged movies (human labels):

| Metric | Baseline (TF-IDF) | Alternative (Semantic Search 1) |
|---|---|---|
| Precision@5 | 0.322 | **0.461** |
| MRR@5 | 0.456 | **0.638** |
| nDCG@5 (judged pool) | 0.481 | **0.631** |

Semantic search is better on 14 queries, worse on 6, with 3 ties. The P@5 gain is +0.14 (bootstrap 95% CI 0.03–0.25, Wilcoxon p = 0.045). Plot alone (nDCG@5 0.657 on 27 queries) beats subtitles alone (0.587–0.590), so dialogue helps less than we expected. Source: `evaluation/results/comparison/comparison_metrics.csv`, `comparison_per_query.csv`.

**With Semantic Search v2** (pooled label set `evaluation/mood_judgments_v2.csv`, AI-assisted first pass; 78 % exact agreement with the human labels on 394 shared pairs, Cohen's κ 0.58), 25 queries:

| Metric | TF-IDF | Semantic Search 1 | **Semantic Search v2** |
|---|---|---|---|
| Precision@5 | 0.336 | 0.488 | **0.568** |
| MRR@5 | 0.449 | 0.707 | **0.745** |
| nDCG@5 (ideal from all judged movies) | 0.255 | 0.434 | **0.504** |

v2 beats TF-IDF on 19 of 24 queries (nDCG +0.26, CI +0.15…+0.37); the gain over v1 (+0.07, CI −0.05…+0.20) is not yet significant. Full tables with intervals, categories and error analysis: `docs/semantic_v2_results.md`.

### Example recommendation

**Input:** "A warm romance about two people finding each other unexpectedly." (M09)

**Output:** Semantic Search 1: *Perfect Match* (grade 2), *Extreme Dating* (1), then three grade-0 dramas. TF-IDF: *Dangerous Love Affairs*, *Bhoot* (a horror film), *Justine and the Whip*, *Born to Kill*, *Salt and Fire*, all graded 0.

**Explanation:** the embedding compares the meaning of the whole request with the plot and dialogue. TF-IDF ranks films that merely share single lemmatised words with the query.

### Known failure

**Input:** "I want a cozy, comforting movie for a rainy evening." (M02)

**Incorrect result:** Semantic Search 1 ranks *Ways to Live Forever*, a drama about a boy with leukaemia, first (grade 0).

**Why it failed:** its dialogue talks about horror films, fear, death and feelings ("you wanted horror films … we kept expecting monsters"), so one of its chunks is close to almost any mood query. It appears in 14 of 27 top-5 lists and is irrelevant in 10 of them. Subtitle similarity measures what characters talk about, not how the film feels.

A second failure shared by TF-IDF and v1 is negation: "No horror, just tension and psychological mystery." (M23) gets 0 relevant movies from both; v2's negation parser gets 4 of 5 relevant (but v2 was designed after seeing this failure). More cases: `docs/failure-journal.md`, `evaluation/results/*_failure_candidates.csv`.

## Project Structure

- `src/` — Python implementation
  - data collection and cleaning: `extract_opus.py`, `extract_opus_metadata.py`, `aligment.py`, `enrich_tmdb_details.py`, `preprocessing.py`
  - TF-IDF baseline: `baseline.py`, `tfidf_text.py`
  - Semantic Search 1: `semantic.py`, `run_semantic_eval.py`
  - Semantic Search v2: `semantic_v2/`, with `configs/v2_default.json` and `tests/`
- `app.py` — Streamlit demo app on top of Semantic Search v2 (`requirements-app.txt`)
- `notebooks/` — Experiments and analysis (`02` TF-IDF evaluation, `03` Colab GPU run, `04` semantic evaluation, `data_exploration`)
- `data/processed/sample_clean_movies.csv` — 100-movie sample (`;`-separated)
- `evaluation/` — mood queries, human judgments, and `results/` with rankings, metrics and failure candidates
- `models/` — saved sample indexes (`tfidf_sample`, `semantic_sample`), so nothing needs re-training to inspect results
- `docs/report.md` — Prefinal report
- `docs/failure-journal.md` — Experiment and failure diary
- `docs/contributions.md` — Team contributions (half page; details in `docs/contributions_detailed.md`)
- `docs/semantic_v2_results.md` — Generated evaluation of all methods (metrics, intervals, error analysis)
- `docs/tfidf_pipeline.md`, `docs/semantic_pipeline.md`, `docs/semantic_v2.md` — how each approach works and how to run it

## How to Run

From the repository root, with the 100-movie sample:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_sm

# TF-IDF baseline (index already saved in models/tfidf_sample)
python -m src.baseline search --index models/tfidf_sample --query "I feel like watching something gloomy and thrilling" --top-k 5

# Semantic Search 1 (index already saved in models/semantic_sample; downloads MiniLM on first use)
python -m src.semantic search --index models/semantic_sample --query "I feel like watching something gloomy and thrilling"

# Recompute the comparison table from the human labels
python -m src.run_semantic_eval metrics
```

To re-run the evaluations, open `notebooks/02_mood_query_evaluation.ipynb` (TF-IDF) and `notebooks/04_semantic_mood_evaluation.ipynb` (semantic). Semantic v2 on the sample, all metrics in one go: `scripts/run_sample_eval.ps1`. Demo app: `py -m pip install -r requirements-app.txt` then `py -m streamlit run app.py` (set `MOVIE_INDEX=models/v2_sample_bge` and `MOVIE_DATA=data/processed/sample_clean_movies.csv` to use the sample index in the repository). Training, full-dataset runs (Colab GPU via `notebooks/03_semantic_search_colab.ipynb`) and troubleshooting are in the docs listed above. Semantic v2: see `docs/semantic_v2.md`.

## Project Status

Prefinal (26 November 2026, tag `prefinal`). **Finished:** data pipeline on the full 51,616-movie corpus; TF-IDF baseline, Semantic Search 1 and Semantic Search v2, all evaluated on the 100-movie sample; v2 full-corpus index (2.25 M subtitle chunks) built and run on 78 queries (unlabelled); Streamlit demo app on v2. **Not done yet:** tuning v2's weights on the labelled dev set, labels on the full corpus, a second human judge, the v2 ablation ladder, app screenshots.

Future work: a calibrated "no good match" answer, better negation and misspelling handling, down-weighting the subtitle channel unless the full-corpus ablation shows a gain, and approximate nearest-neighbour search (median 4.5 s per query on the full index today).
