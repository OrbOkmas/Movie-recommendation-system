# Movie Recommendation System

## Project Overview

This project is developed as part of the Text Analytics 2 course at Prague University of Economics and Business.

Our goal is to develop and evaluate a movie recommendation system using natural language processing techniques.

## Team Members

| Member | Responsibility |
|---|---|
| [Name 1] | Project coordination and integration |
| [Name 2] | Dataset preparation and preprocessing |
| [Name 3] | Baseline recommendation approach |
| [Name 4] | Semantic recommendation approach |
| [Name 5] | Evaluation and error analysis |
| [Name 6] | Documentation and application development |

## Project Objectives

- Prepare and explore a movie dataset.
- Implement at least two recommendation approaches.
- Compare their performance using suitable evaluation metrics.
- Analyse successful recommendations and failure cases.
- Discuss practical limitations and improvements.

## Dataset

**Source:** [Dataset name and URL]

**Description:** [What information is available?]

**Data preparation:** [Cleaning and preprocessing steps]

## Methodology

### Approach 1: TF-IDF Baseline and Mood Evaluation

The lexical movie recommendation baseline is implemented in `src/baseline.py`, with consistent query preprocessing in `src/tfidf_text.py`.

The notebook `notebooks/02_mood_query_evaluation.ipynb` runs the shared mood queries, exports ranked recommendations, and supports human relevance judgments and evaluation metrics.

For setup instructions, training and search commands, evaluation procedures, full-dataset execution, and troubleshooting, see the [TF-IDF Pipeline Usage Guide](docs/tfidf_pipeline.md).

### Approach 2: Semantic Search 1 (sentence embeddings with subtitle chunking)

Implemented in `src/semantic.py`, evaluated in `notebooks/04_semantic_mood_evaluation.ipynb`.

Movies and queries are encoded with the pre-trained sentence-embedding model `all-MiniLM-L6-v2`, so a query can match a movie by meaning even without shared words. Because the model reads only ~200 words, each movie gets one vector for its metadata + plot and one vector per 150-word subtitle chunk (≈ 42 chunks per movie). A movie's score combines the plot similarity and its best-matching subtitle chunks (0.5 / 0.5); every recommendation comes with the dialogue passage that matched the query, which makes the results explainable. Unlike TF-IDF, the query is not stop-word filtered or lemmatised.

Details, results and how to run it: [Semantic Search 1 documentation](docs/semantic_pipeline.md).

## Results

Human relevance judgments (0/1/2) on the 100-movie sample, 23 mood queries judged for both methods:

| Metric | TF-IDF baseline | Semantic Search 1 |
|---|---:|---:|
| Precision@5 | 0.322 | **0.461** |
| MRR@5 | 0.456 | **0.638** |
| nDCG@5 (judged pool) | 0.481 | **0.631** |

All results were judged by the same team member; see the documentation for limitations.

### Example recommendation

**Input:** [Example user preference]

**Output:** [Actual recommendations]

**Explanation:** [Why these results are relevant]

### Known failure

**Input:** [Example]

**Incorrect result:** [Output]

**Why it failed:** [Explanation]

## Project Structure

- `src/` — Python implementation
- `notebooks/` — Experiments and analysis
- `docs/report.md` — Prefinal report
- `docs/failure-journal.md` — Experiment and failure diary
- `docs/contributions.md` — Team contributions
- `docs/results.md` — Detailed evaluation

## How to Run

[Add installation and execution instructions once the application is working.]

## Project Status

In development.
