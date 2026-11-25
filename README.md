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

### Approach 2 — Alternative
[Describe the second approach and how it differs.]

## Results

| Metric | Baseline | Alternative |
|---|---|---|
| [Metric 1] | TBD | TBD |
| [Metric 2] | TBD | TBD |

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

### Semantic search v2 (Matyáš)

Improved semantic retrieval: separate plot/keyword/genre vectors, calibrated fusion, hub correction, negation and typo handling, rule-based abstention. Usage and design: `docs/semantic_v2.md`, `docs/semantic_v2_plan.md`; results: `docs/semantic_v2_results.md`; run everything on the sample with `scripts/run_sample_eval.ps1`.
