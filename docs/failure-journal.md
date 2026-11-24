
# Failure Journal

Project: Movie Recommendation System

## Purpose

This journal documents experiments, unsuccessful approaches, errors, and lessons learned throughout our project.

## Experiment Log

### Experiment 1 — [Date]

**Team member:**

**What we tried:**

**Why we tried it:**

**What went wrong:**

**Evidence / Example:**

**What we changed:**

**What we learned:**

## Experiment Log

### Experiment 2 [19.11.2026]— TF-IDF Baseline: Query Processing and Retrieval Limitations

**Team member: Eliška**

**What we tried:**

We implemented a TF-IDF movie recommendation baseline using preprocessed movie metadata, plots, and subtitles. We trained the model on a sample of 100 movies and tested it with 28 natural-language mood queries. We also implemented query preprocessing to remove conversational filler, stop words, and normalize words using lemmatization.

**Why we tried it:**

We wanted to establish a simple lexical baseline for mood-based movie recommendations. This would allow us to identify the strengths and weaknesses of TF-IDF before comparing it with semantic search and a hybrid approach.

**What went wrong:**

- **Missing recommendations:** Three queries (M25, M26, M28) produced no results because they were too vague, contained misspellings, or were empty.
- **Loss of negation:** Stop-word removal eliminates important words such as "not", potentially reversing the user's intended meaning.
- **Poor mood matching:** Some recommended movies did not appear to match the requested atmosphere, demonstrating the limitations of keyword-based similarity.
- **Document truncation:** The baseline was configured to use a maximum of 4,000 words per movie to control memory usage and computational cost. This removed potentially useful text from longer documents.

**Evidence / Example:**

- **M25:** "I feel like watching something." → No useful query terms remained after preprocessing.
- **M26:** "gloooomy thriling mistree" → Misspelled words did not match the TF-IDF vocabulary.
- **M28:** Empty query → Rejected by input validation.
- **Negation example:** "I want something dark but not scary" → Removing "not" changes the intended meaning.
- **M01:** "I feel like watching something gloomy and thrilling" → Returned three movies, including comedies that appeared poorly matched to the requested mood.
- **Training warning:** 7 out of 100 movie documents were truncated to 4,000 words.

**What we changed:**

We implemented consistent query preprocessing and removal of conversational filler. We also included edge-case queries in our evaluation set to identify failure cases systematically.

We documented the limitations of stop-word removal and document truncation. These issues have not yet been fully resolved. For future experiments, we plan to investigate preserving negation, handling spelling errors, and increasing the document-length limit.

**What we learned:**

TF-IDF provides a useful and interpretable baseline, but it relies on vocabulary overlap rather than a deeper understanding of meaning. Preprocessing decisions can remove important information, particularly negation, while restricting document length may discard relevant subtitle content. We also learned that empty, vague, and misspelled queries require special handling. These findings motivate our planned comparison with semantic retrieval and evaluation on the full movie dataset.
