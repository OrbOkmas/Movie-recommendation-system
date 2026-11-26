# Movie Recommendation System — Prefinal Report

## 0. Goal and Results

**Project goal:**  
Develop and evaluate a movie recommendation system that helps users discover relevant movies based on their preferences.

**Research question:**  
[Specify the question your team is investigating, such as whether semantic similarity improves recommendation relevance compared with a simpler baseline.]

**Current status:**  
[Describe what is implemented, what is still in progress, and what is only planned or mocked.]

**Main results:**  
[Include numerical metrics, key findings, and an example recommendation once available.]

## 1. Data and Methodology

### Dataset
- **Source:** OPUS OpenSubtitles `v2018` English corpus (`en.zip`, 24 GB; Lison & Tiedemann, 2016) aligned by IMDb identifier with metadata from The Movie Database (TMDB API v3/v4).
- **Size:** **51,616** verified feature films (filtered down from 446,612 raw XML subtitle files and 140,044 unique IMDb IDs after removing duplicate uploads and TV series episodes), stored in `data/processed/clean_movies.parquet` alongside a 100-movie evaluation sample (`data/processed/sample_clean_movies.csv`).
- **Features:** Unique cross-platform identifiers (`opus_id`, `imdb_id`, `tmdb_id`), `tmdb_title`, `release_year` (1895–2018 corpus cutoff), `genres` (pipe-delimited), `keywords` (pipe-delimited), `plot` synopsis, `poster_url`, `original_language`, `runtime`, `vote_average`, `vote_count`, cleaned subtitle dialogue (`clean_subtitles`), and diagnostic length/quality flags (`plot_word_count`, `subtitle_word_count`, `has_valid_subtitles`).
- **Data quality:** Primary keys (`opus_id`, `imdb_id`, `tmdb_id`) have **0** duplicates. Missing rates are low: `0.52%` (270 movies) lack a TMDB plot, `27.64%` (14,265 movies) lack TMDB keywords, `0.18%` (92 movies) have incomplete subtitles (`< 50` words), and `0.07%` (34 movies) exceed `30,000` subtitle words. Because the corpus originates from the English OPUS archive (`en.zip`) with a 2018 cutoff, it skews toward post-2000 releases and Anglophone productions.

### Preprocessing
Implemented as a reproducible, streaming pipeline in `src/preprocessing.py` (with exploratory diagnostics in `notebooks/01_data_exploration.ipynb`) that produces two distinct text representations tailored to downstream model families:
1. **Shared Stage 1 Cleaning & Detokenization:** Streams `raw_subtitles.txt` line-by-line for the 51,616 matched films, normalizes Unicode (`NFKC`), strips XML/HTML tags, hearing-impaired brackets (`[Music]`, `(Sighs)`), musical symbols, speaker hyphens, and uploader watermarks, and restores natural punctuation/contraction spacing split by OPUS `<w>` token joins.
2. **Quality-Flag Fallback (`has_valid_subtitles`):** Movies with fewer than 50 cleaned subtitle words (92 records) are flagged (`has_valid_subtitles = False`) and omit dialogue from both downstream text representations, falling back cleanly to `Title + Genres + Keywords + Plot`.
3. **Track A — Lexical Input (`text_tfidf`):** Combines title, genres, keywords, plot, and valid dialogue (capped at 10,000 words per document to prevent extreme-length outliers from skewing term weights), then applies lowercasing, `spaCy` tokenization, English stop-word removal, and `WordNet` lemmatization.
4. **Track B — Dense Semantic Input (`text_embedding`):** Preserves natural casing, punctuation, and sentence boundaries in a structured template (`Title: ... | Genres: ... | Keywords: ... | Plot: ... | Dialogue: ...`) for transformer encoders (`all-MiniLM-L6-v2`, `BGE`).

## TF-IDF Baseline

### Objective and Dataset

We implemented a Term Frequency–Inverse Document Frequency (TF-IDF) retrieval model as the lexical baseline for our mood-based movie recommendation system. The goal was to evaluate how well a relatively simple, interpretable method can translate natural-language requests, such as “something gloomy and thrilling,” into relevant movie recommendations. This baseline provides a reference point for subsequent semantic and emotion-aware methods.

The experiment used the available **100-movie sample** from our preprocessed movie dataset. The sample consists of the first 100 records with valid subtitles selected by the data-preparation pipeline; it is not a random or representative sample of the full corpus. Each movie's `text_tfidf` field combines its title, genres, keywords, plot description and cleaned subtitle dialogue. The movie texts were prepared using lowercasing, spaCy tokenization, stop-word removal and WordNet lemmatization.

### Model Implementation

The model is implemented in `src/baseline.py`, with query preprocessing separated into `src/tfidf_text.py`. The TF-IDF vectorizer uses unigrams and bigrams, a maximum vocabulary of 50,000 features, sublinear term-frequency scaling and L2 normalization.

During training, each movie is represented as a sparse TF-IDF vector. For a new query, the fitted vectorizer transforms the processed query into the same feature space. Movies are ranked using the dot product of L2-normalized vectors, equivalent to cosine similarity. The trained vocabulary, sparse movie matrix and metadata are saved for reuse, so the model does not require retraining for each query.

Query preprocessing removes conversational expressions such as “I feel like watching” and then applies tokenization, stop-word removal and lemmatization. For example, “I feel like watching something gloomy and thrilling” becomes approximately `gloomy thrilling`.

To control memory consumption and processing time, we initially restricted each movie document to 4,000 words. All 100 movies were successfully indexed, with no invalid records dropped, although seven movie documents were truncated. This restriction is an implementation decision rather than an inherent limitation of TF-IDF.

### Evaluation Methodology

We evaluated the model using **28 predefined natural-language mood queries** covering direct emotions, mood-related themes, subtle atmospheric descriptions, negation and edge cases.

The evaluation notebook (`02_mood_query_evaluation.ipynb`) generated ranked recommendations and prepared a human evaluation sheet. Recommendations were manually assessed using a three-level relevance scale:

- **0:** Irrelevant — the movie does not match the requested mood.
- **1:** Partially relevant — the movie matches some aspects of the request.
- **2:** Highly relevant — the movie strongly matches the requested atmosphere or theme.

We calculated three retrieval metrics: Precision@5, Mean Reciprocal Rank (MRR@5), and normalized Discounted Cumulative Gain (nDCG@5). Relevance grades of 1 or 2 were treated as relevant for Precision and MRR, while nDCG incorporated the graded relevance values. The implemented nDCG uses the judged retrieved pool to construct its ideal ranking.

### Results

The baseline returned **119 movie recommendations across 25 of the 28 queries**. Twenty-three queries returned five recommendations, while two returned fewer than five. Two queries produced no matches, and one empty query was rejected by input validation.

The quantitative evaluation produced the following results:

| Evaluation metric | Mean score |
|---|---:|
| Precision@5 | **0.322** |
| MRR@5 | **0.456** |
| nDCG@5 (judged pool) | **0.481** |

*Results are averaged over the 23 queries that returned five movies and had complete human relevance judgments.*

The results indicate moderate but inconsistent retrieval quality. Seven evaluated queries (M09, M11, M15, M17, M21, M23 and M24) returned no human-relevant movies in their top five.

The model struggled particularly with some abstract atmospheric descriptions, such as “a dreamy and surreal movie with a beautiful but disturbing atmosphere” (M17), which achieved Precision@5 = 0.

However, TF-IDF performed better for certain requests. For example, “I want to feel uneasy and curious at the same time” (M16) achieved Precision@5 = 0.8. This indicates that lexical retrieval can be useful when the query vocabulary overlaps meaningfully with movie descriptions.

### Limitations and Failure Analysis

**1. Limited understanding of emotions and atmosphere**

TF-IDF relies on lexical overlap rather than a deeper interpretation of emotional meaning. Consequently, a movie may receive a high similarity score because relevant words occur in its plot or dialogue, even when its overall atmosphere differs from the user's request.

**2. Loss of negation**

Our preprocessing removes common stop words, including expressions such as “not”. This can change the intended meaning of a query.

For example, “No horror, just tension and psychological mystery” is reduced to `horror tension psychological mystery`. The model consequently treats “horror” as a desired term rather than an excluded characteristic.

This query (M23) achieved Precision@5 = 0. However, some other queries containing negation achieved higher scores. This suggests that TF-IDF may occasionally produce relevant results despite losing negation, rather than genuinely understanding negative preferences.

**3. Sensitivity to vague and misspelled queries**

M25 (“I feel like watching something”) contained no useful vocabulary after preprocessing. M26 (“gloooomy thriling mistree”) contained misspelled words that did not match the model's learned vocabulary. Both produced no recommendations. The empty query M28 was rejected by input validation.

**4. Document truncation**

Seven movies exceeded the configured 4,000-word limit. Truncation may remove useful mood-related information from subtitles, particularly dialogue appearing later in the film. The impact of this restriction has not yet been measured.

**5. Limited evaluation coverage**

The reported metric averages cover only 23 of the 28 queries. M01 and M05 were excluded because they returned fewer than five recommendations, while M25, M26 and M28 produced no valid full ranking. Consequently, the reported metrics may overestimate overall usefulness by excluding some difficult cases.

Additionally, the small, non-random 100-movie sample limits the generalizability of the results.

### Conclusion and Next Steps

The TF-IDF baseline established a reproducible lexical retrieval approach and revealed important limitations in mood-based movie recommendation. Although the method successfully retrieved relevant films for some queries, it struggled with complex emotional descriptions, negation and vocabulary mismatches.

These findings motivate our comparison with semantic and emotion-aware recommendation methods, which may better represent the atmosphere and emotional characteristics of films.

For future experiments, we plan to evaluate the baseline on the full movie dataset, investigate explicit negative-preference handling, test different document-length restrictions and improve retrieval coverage reporting. All recommendation approaches will be compared using the same movie corpus, evaluation queries and human relevance judgments.

**Approach B — Alternative**  
[Describe your second implemented approach, such as sentence embeddings.]

Explain why the team chose these methods and how each one works.

## 2. Comparison of Approaches and Error Analysis

### Evaluation

| Metric | Approach A | Approach B |
|---|---|---|
| [Metric 1] | [Result] | [Result] |
| [Metric 2] | [Result] | [Result] |
| [Metric 3] | [Result] | [Result] |

### Concrete examples

**Successful example**
- Input: [User query or selected movie]
- Output: [Recommendation]
- Explanation: [Why the result is appropriate]

**Failure example**
- Input: [Problematic case]
- Output: [Incorrect or irrelevant recommendation]
- Explanation: [Why the system failed]

### Interpretation

[What does the comparison demonstrate? What alternative explanations might account for the results?]

## 3. Limits and Risks

Discuss limitations supported by your experiments, such as:
- Dataset coverage and biases
- Cold-start problems
- Ambiguous preferences
- Popularity bias
- Computational limitations

**What we would improve next:**  
[Concrete improvements justified by the evidence.]

**Repository:** [Insert public GitHub URL]

**Team members:** [Names]
