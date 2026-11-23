# Report text — Semantic Search 1 (for `docs/report.md`)

> Ready-to-paste text for the prefinal report. Part A goes to **1. Data and Methodology → Approach B**, Part B to **2. Comparison of Approaches and Error Analysis**, Part C to **3. Limits and Risks**. Kept in a separate file so that report edits by other team members do not collide in a merge.

---

## Part A — Approach B: Semantic Search 1 (sentence embeddings)

### Objective

The TF-IDF baseline can only match words, but mood requests are paraphrastic: "gloomy", "dark" and "bleak" describe similar feelings, and a request rarely uses the same words as a plot or a dialogue. Semantic Search 1 tests whether ranking movies by **meaning** rather than by shared vocabulary retrieves more relevant films, and whether **subtitles** add mood information beyond the plot description.

### Model Implementation

We use the pre-trained sentence-embedding model `all-MiniLM-L6-v2` (sentence-transformers; 384-dimensional vectors). It was chosen as a standard, fast encoder that runs on a laptop CPU and needs no training on our data. Movies and queries are mapped into the same vector space and ranked by cosine similarity. The implementation is in `src/semantic.py`.

The model reads at most 256 word-pieces (about 150–200 words) and silently truncates longer input, while subtitles in our sample contain on average 6,200 words per movie. Encoding a whole movie as one vector would therefore cover less than 2 % of its dialogue. We split each movie's `text_embedding` field into (1) **metadata and plot** — one vector per movie — and (2) **dialogue**, cut into 150-word chunks (4,155 chunks, 41.6 per movie; a trailing chunk shorter than 50 words is merged into the previous one). A movie's subtitle score aggregates the similarities of its chunks (best chunk, or mean of the three best), and the final score combines both parts equally:

`score = 0.5 · cos(query, plot) + 0.5 · mean of the 3 best cos(query, subtitle chunk)`

The query is encoded as written, without stop-word removal or lemmatisation, so negations such as "not" reach the model. Each recommendation is returned with the subtitle passage that matched the query best, which makes the ranking explainable. All variants are computed from one saved index (encoding the sample took 152 s on a laptop CPU), so different scoring rules can be compared without re-encoding.

### Evaluation Methodology

We used the same 100-movie sample, the same 28 mood queries, the same 0/1/2 relevance scale and the same metric implementation as for TF-IDF (`notebooks/04_semantic_mood_evaluation.ipynb`). Because we tested four semantic variants against one baseline, we fixed the **main comparison before labelling**: TF-IDF vs `sem_combined`, which uses the same information (plot + subtitles). The three other variants — plot only, subtitles with best-chunk aggregation, subtitles with top-3 aggregation — are reported as an ablation.

Labels were collected in a pooled, blind sheet: each query–movie pair appears once regardless of how many methods retrieved it, and the sheet does not show which method recommended a movie. The 33 semantic pairs already judged for TF-IDF reused their grades; the remaining 275 pairs were judged on 23 November by the same team member who judged the TF-IDF results.

---

## Part B — Comparison of Approaches and Error Analysis

### Evaluation

Main comparison on the 23 queries for which both methods returned five judged movies:

| Metric | TF-IDF (Approach A) | Semantic Search 1 (Approach B) |
|---|---:|---:|
| Precision@5 | 0.322 | **0.461** |
| MRR@5 | 0.456 | **0.638** |
| nDCG@5 (judged pool) | 0.481 | **0.631** |

Semantic Search 1 is better on 14 queries, TF-IDF on 6, with 3 ties (nDCG@5). The difference in Precision@5 is +0.14 (bootstrap 95 % CI 0.03–0.25; Wilcoxon signed-rank p = 0.045), in nDCG@5 +0.15 (CI 0.03–0.28; p = 0.07). The improvement is consistent but rests on 23 queries.

Ablation of the text source (all 27 non-empty queries):

| Semantic variant | Precision@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|
| plot + subtitles (`sem_combined`) | **0.467** | 0.630 | 0.641 |
| plot only | 0.422 | **0.631** | **0.657** |
| subtitles only, top-3 chunks | 0.422 | 0.548 | 0.590 |
| subtitles only, best chunk | 0.407 | 0.504 | 0.587 |

### Concrete examples

**Successful example**
- Input: "A warm romance about two people finding each other unexpectedly." (M09)
- Output (semantic): *Perfect Match* (grade 2; the matched subtitle chunk is a love-song lyric), *Extreme Dating* (1), then three grade-0 dramas.
- TF-IDF returned *Dangerous Love Affairs*, the horror film *Bhoot*, *Justine and the Whip*, *Born to Kill* and *Salt and Fire* — all graded 0.
- Explanation: the embedding compares the meaning of the whole request with the plot and dialogue, while TF-IDF ranks films that merely share individual (lemmatised) words with the query, regardless of the context in which they occur.

**Failure example 1 — a "hub" movie**
- Input: "I want a cozy, comforting movie for a rainy evening." (M02)
- Output: rank 1 *Ways to Live Forever*, a drama about a boy with leukaemia (grade 0).
- Explanation: its dialogue discusses horror films, fear, death and feelings, so at least one chunk is similar to almost any mood query. It appears in 14 of 27 top-5 lists and is irrelevant in 10 of them. Subtitle similarity measures what characters *talk about*, not how the film *feels*.

**Failure example 2 — negation**
- Input: "No horror, just tension and psychological mystery." (M23)
- Output: horror films such as *Scared*, *Bhoot* and *Crypt of the Vampire*; 0 relevant movies, the same as TF-IDF.
- Explanation: the sentence encoder places "no horror" close to "horror". Keeping "not/no" in the query, which TF-IDF removes, did not help.

### Interpretation

Ranking by meaning improves retrieval over word overlap on our sample, mainly for direct moods (Precision@5 0.70 vs 0.30) and mood themes (0.43 vs 0.23); the gain is small for subtle atmospheric queries and negation, and TF-IDF is better for the vague query M27. The ablation shows that the improvement comes mostly from the embedding of the **plot**: subtitles alone perform worse than the plot alone and add only a little in combination, because dialogue often talks *about* emotions and genres instead of expressing the film's atmosphere. Alternative explanations: the two sheets were labelled in separate sessions and the second one contained only semantic results, so the judge could tell which method they came from; both methods were also evaluated on a small, non-random sample.

---

## Part C — Limits and Risks (Semantic Search 1)

- **Single judge, two sessions:** all labels come from one person, so they are consistent but subjective; a second judge on a sample of pairs would quantify agreement. The semantic sheet was labelled later and contained only semantic results, so blinding between methods was only partial.
- **Hub movies:** a few movies with emotionally varied dialogue dominate the rankings; a per-movie score normalisation is planned.
- **Negation and misspellings** remain unsolved; a spelling-correction pre-processing step was tested and not adopted (journal, Experiment 8).
- **Noisy subtitles:** translator credits and song lyrics are encoded as dialogue.
- **Always returns results:** dense retrieval returns five movies even for meaningless input; a minimum-similarity threshold would be needed for a "no good match" answer.
- **Scale:** the full dataset (~50,000 movies, ~2 million chunks) needs a GPU (≈ 21 h estimated on a laptop CPU); all results above are from the 100-movie sample.
