# Movie Recommendation System

## Project Overview

This project is developed as part of the Text Analytics 2 course at Prague University of Economics and Business.

Our goal is to develop and evaluate a movie recommendation system using natural language processing techniques. The user describes a **mood** in free text ("something gloomy and thrilling", "no horror, just tension") and the system returns five matching movies. We test whether ranking by meaning beats ranking by shared words, and whether subtitle dialogue adds mood information beyond the plot description.

## Team Members

| Member | Responsibility |
|---|---|
| [?] | Project coordination and integration |
| [?] | Dataset preparation and preprocessing |
| Eliška [?] | Baseline recommendation approach (TF-IDF) |
| Tomáš | Semantic recommendation approach (Semantic Search 1) |
| [?] | Evaluation and error analysis |
| [?] | Documentation and application development |

Also: Matyáš — Semantic Search v2 (in progress); Vojta — full-dataset runs [?]. Details in `docs/contributions.md`.

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

**In progress, Semantic Search v2** (`src/semantic_v2/`, [docs](docs/semantic_v2.md)): separate plot, keyword and genre vectors, a correction for "hub" movies, a negation penalty, spelling correction and an abstention status. It is implemented and has offline tests, but its results are not labelled yet.

**Evaluation (both approaches):** the same 28 mood queries (`evaluation/mood_queries.json`; direct moods, mood themes, subtle vibes, negation, edge cases), one judge grading each query/movie pair 0 / 1 / 2 in pooled sheets (`evaluation/mood_judgment_tfidf.csv`, `evaluation/mood_judgment_semantic.csv`), and Precision@5, MRR@5 and nDCG@5.

## Results

On the 23 queries where both methods returned five judged movies:

| Metric | Baseline (TF-IDF) | Alternative (Semantic Search 1) |
|---|---|---|
| Precision@5 | 0.322 | **0.461** |
| MRR@5 | 0.456 | **0.638** |
| nDCG@5 (judged pool) | 0.481 | **0.631** |

Semantic search is better on 14 queries, worse on 6, with 3 ties. The P@5 gain is +0.14 (bootstrap 95% CI 0.03–0.25, Wilcoxon p = 0.045). Plot alone (nDCG@5 0.657 on 27 queries) beats subtitles alone (0.587–0.590), so dialogue helps less than we expected. Source: `evaluation/results/comparison/comparison_metrics.csv`, `comparison_per_query.csv`.

### Example recommendation

**Input:** "A warm romance about two people finding each other unexpectedly." (M09)

**Output:** Semantic Search 1: *Perfect Match* (grade 2), *Extreme Dating* (1), then three grade-0 dramas. TF-IDF: *Dangerous Love Affairs*, *Bhoot* (a horror film), *Justine and the Whip*, *Born to Kill*, *Salt and Fire*, all graded 0.

**Explanation:** the embedding compares the meaning of the whole request with the plot and dialogue. TF-IDF ranks films that merely share single lemmatised words with the query.

### Known failure

**Input:** "I want a cozy, comforting movie for a rainy evening." (M02)

**Incorrect result:** Semantic Search 1 ranks *Ways to Live Forever*, a drama about a boy with leukaemia, first (grade 0).

**Why it failed:** its dialogue talks about horror films, fear, death and feelings ("you wanted horror films … we kept expecting monsters"), so one of its chunks is close to almost any mood query. It appears in 14 of 27 top-5 lists and is irrelevant in 10 of them. Subtitle similarity measures what characters talk about, not how the film feels.

A second failure shared by both methods is negation: "No horror, just tension and psychological mystery." (M23) gets 0 relevant movies from both. More cases: `docs/failure-journal.md`, `evaluation/results/*_failure_candidates.csv`.

## Project Structure

- `src/` — Python implementation
  - data collection and cleaning: `extract_opus.py`, `extract_opus_metadata.py`, `aligment.py`, `enrich_tmdb_details.py`, `preprocessing.py`
  - TF-IDF baseline: `baseline.py`, `tfidf_text.py`
  - Semantic Search 1: `semantic.py`, `run_semantic_eval.py`
  - Semantic Search v2 (in progress): `semantic_v2/`, with `configs/v2_default.json` and `tests/`
- `notebooks/` — Experiments and analysis (`02` TF-IDF evaluation, `03` Colab GPU run, `04` semantic evaluation, `data_exploration`)
- `data/processed/sample_clean_movies.csv` — 100-movie sample (`;`-separated)
- `evaluation/` — mood queries, human judgments, and `results/` with rankings, metrics and failure candidates
- `models/` — saved sample indexes (`tfidf_sample`, `semantic_sample`), so nothing needs re-training to inspect results
- `docs/report.md` — Prefinal report
- `docs/failure-journal.md` — Experiment and failure diary
- `docs/contributions.md` — Team contributions
- `docs/results.md` — Detailed evaluation
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

To re-run the evaluations, open `notebooks/02_mood_query_evaluation.ipynb` (TF-IDF) and `notebooks/04_semantic_mood_evaluation.ipynb` (semantic). Training, full-dataset runs (Colab GPU via `notebooks/03_semantic_search_colab.ipynb`) and troubleshooting are in the docs listed above. Semantic v2: see `docs/semantic_v2.md`.

## Project Status

Prefinal (26 November 2026). Finished: data pipeline, TF-IDF baseline, Semantic Search 1, and their labelled comparison on the 100-movie sample. In progress: Semantic Search v2 (implemented and tested, not yet labelled). Not done yet: runs and labels on the full 51,616-movie corpus, and a second judge to measure agreement.

Future work: a hybrid that adds emotion tagging to semantic retrieval [?], a calibrated "no good match" answer for vague or nonsense queries, and better handling of negation and misspellings.
