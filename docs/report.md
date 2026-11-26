# Mood-Based Movie Recommendation — Prefinal Report

**Repository:** <https://github.com/OrbOkmas/Movie-recommendation-system> (tag `prefinal`)  
**Team:** Eliška, Matouš, Matyáš, Sam, Tomáš, Vojta · Text Analytics 2, Prague University of Economics and Business · 26 November 2026

## 0. Goal and Results

**Goal.** A user describes a *mood* in free text ("something gloomy and thrilling", "no horror, just tension") and gets five matching movies. We test two questions: (1) does ranking by **meaning** (sentence embeddings) beat ranking by **shared words** (TF-IDF)? (2) does **subtitle dialogue** carry mood information beyond the plot summary?

**Main results** (100-movie sample, 28 mood queries, human/AI-assisted 0/1/2 relevance labels, top 5):

| Method | Precision@5 | MRR@5 | nDCG@5 (pool) | Label set |
|---|---:|---:|---:|---|
| TF-IDF baseline | 0.322 | 0.456 | — | human, 23 queries |
| Semantic Search v1 (MiniLM, plot + subtitles) | 0.461 | 0.638 | — | human, 23 queries |
| TF-IDF baseline | 0.336 | 0.449 | 0.255 | v2 pooled, 25 queries |
| Semantic Search v1 | 0.488 | 0.707 | 0.434 | v2 pooled, 25 queries |
| **Semantic Search v2** (BGE, field fusion, negation) | **0.568** | **0.745** | **0.504** | v2 pooled, 25 queries |

- Meaning beats words: v1 > TF-IDF on 14 of 23 queries (P@5 +0.14, bootstrap 95 % CI 0.03–0.25, Wilcoxon p = 0.045); v2 > TF-IDF on 19 of 24 (nDCG +0.26, CI +0.15…+0.37).
- **Subtitles help less than we expected**: plot-only embeddings (nDCG@5 0.657) beat subtitles-only (0.587–0.590). In v2, the subtitle similarity is the same for relevant and irrelevant results (0.557 vs 0.554).
- v2 vs v1 is a gain we **cannot yet prove**: nDCG +0.07, CI −0.05…+0.20.

**Sample run (v2):** "No horror, just tension and psychological mystery." (M23) → *The Uninvited* (1), *The Nameless* (1), *A Shadow of a Doubt* (1), *The Headless Woman* (2), *Des Pardes* (0). TF-IDF and v1 return horror films (*Scared*, *Bhoot*, *Crypt of the Vampire*) — 0 relevant.

**Status.** *Finished:* data pipeline on the full 51,616-movie corpus; TF-IDF, v1 and v2 implemented and evaluated on the sample; v2 full-corpus index (2.25 M subtitle chunks) built and run on 78 queries; Streamlit demo app on top of v2 (`app.py`). *In progress / not done:* v2 weights are untuned placeholders; no labels on the full corpus; only one human judge (agreement with the AI-assisted labels measured, see §3); v2 ablation ladder.

## 1. Data and Methodology

### Dataset
- **Source:** OPUS OpenSubtitles `v2018` English corpus (`en.zip`, 24 GB; Lison & Tiedemann, 2016) aligned by IMDb id with TMDB API metadata.
- **Size:** **51,616** feature films, filtered from 446,612 raw XML subtitle files / 140,044 unique IMDb ids (duplicate uploads and TV episodes removed); `data/processed/clean_movies.parquet`. All labelled experiments use a 100-movie sample (`data/processed/sample_clean_movies.csv`) — the *first* 100 movies with valid subtitles, **not random**.
- **Per movie:** ids (`opus_id`, `imdb_id`, `tmdb_id`), title, year, genres, keywords, plot, poster, language, runtime, ratings, cleaned dialogue (median 6,205 words) and quality flags.
- **Quality:** 0 duplicate keys. Missing: plot 0.52 % (270), keywords 27.6 % (14,265), genres 1.3 %, dialogue < 50 words 0.18 % (92); 34 movies exceed 30,000 words. Drama 50 %, Comedy 29 %, Horror 11 %; English-language archive, many obscure titles.

### Preprocessing (`src/preprocessing.py`)
1. **Cleaning:** streams the subtitles; NFKC normalisation; strips XML/HTML tags, hearing-impaired brackets (`[Music]`, `(Sighs)`), music symbols, speaker hyphens and uploader watermarks; repairs OPUS token spacing ("don ' t" → "don't"). *Residual noise:* translator credits and song lyrics survive (e.g. the best-matching "dialogue" of *Scared* for M23 is "German subtitles translated by …").
2. **Fallback:** movies with < 50 dialogue words (92) get `has_valid_subtitles = False` and are represented by title + genres + keywords + plot only.
3. **Two representations:** `text_tfidf` (lowercased, spaCy tokens, stop words removed, WordNet lemmas, ≤ 10,000 words) for the lexical model; `text_embedding` (`Title: … | Genres: … | Keywords: … | Plot: … | Dialogue: …`, casing and punctuation kept) for encoders.

### Approach A — TF-IDF baseline (`src/baseline.py`, `src/tfidf_text.py`)
Unigrams + bigrams, 50,000 features, sublinear TF, L2 norm, cosine ranking; documents capped at 4,000 words (7 of 100 truncated). Queries get the same preprocessing plus removal of filler ("I feel like watching") — "I feel like watching something gloomy and thrilling" → `gloomy thrilling`. Chosen as an interpretable reference point: it shows how far plain word overlap gets.

### Approach B — Semantic Search v1 (`src/semantic.py`)
`all-MiniLM-L6-v2` (384-d), queries encoded as written (so "not" reaches the model). MiniLM reads ≈ 200 words, so one vector per movie would see < 2 % of the dialogue; instead each movie gets one **plot + metadata** vector and **150-word dialogue chunks** (41.6 per movie; tails < 50 words merged). Score = 0.5 · cos(plot) + 0.5 · mean of the 3 best chunk cosines; the best chunk is returned as an explanation. The main variant (`sem_combined`) was **fixed before labelling**; plot-only and subtitles-only variants are an ablation.

### Approach C — Semantic Search v2 (`src/semantic_v2/`)
Built to fix v1's documented failures (failure journal, exp. 5–14):

| v1 failure | v2 change |
|---|---|
| one concatenated string; title words dominate | separate vectors for plot, keywords, genres (BGE-base) and chunks (BGE-small); missing field = no score |
| raw cosines of different sources mixed 0.5/0.5 | per-query z-score per source, weighted fusion (plot 1.0, keywords 0.4, genres 0.2, subtitles 0.5) |
| one movie in 14–19 of 27 top-5 lists | **hub correction**: subtract each movie's mean similarity to 800 generic background queries |
| "no horror" ≈ "horror" | **negation parser** + penalty λ·max(0, z_neg − τ) |
| typos, vague input → random films | spell correction (SymSpell + corpus vocabulary) and explicit statuses (`low_confidence`, `negation_only`, `empty`) |

### Evaluation
28 test queries M01–M28 in 5 categories (direct mood, mood + theme, subtle vibe, negation, edge cases: vague, misspelled, empty); 50 separate dev queries for v2 tuning. Each query–movie pair graded 0 (irrelevant) / 1 (partly) / 2 (strongly) in a **pooled, method-blind** sheet. Metrics of different character: Precision@5 (how many hits), MRR@5 (how soon the first hit), nDCG@5 (graded order). We report nDCG with the ideal ranking taken from **all** judged movies of the query ("pool"): the original team version took it from the retrieved list itself, so a method that missed every relevant movie but ranked its few hits well scored highly (failure journal, exp. 7).

## 2. Comparison of Approaches and Error Analysis

### Overall and by query type
Human labels, 23 queries where all methods returned 5 judged films: TF-IDF P@5 0.322 → v1 0.461. Pooled v2 label set, 25 queries (table in §0, full results with intervals in `docs/semantic_v2_results.md`):

| nDCG@5 (pool) by category | TF-IDF | v1 combined | v1 plot only | v2 |
|---|---:|---:|---:|---:|
| direct mood (6) | 0.13 | **0.51** | 0.46 | 0.41 |
| mood + theme (6) | 0.33 | 0.46 | 0.32 | **0.59** |
| subtle vibe (6) | 0.18 | 0.29 | 0.39 | **0.50** |
| negation (6) | 0.26 | 0.37 | 0.41 | **0.42** |

v2 wins on themes and subtle vibes but **loses on direct moods** to v1. The ablation answers question (2): subtitles alone are the weakest semantic source (P@5 0.36–0.38 vs 0.48 plot-only) and add only +0.01 P@5 in combination.

**Label-free diagnostics** (no labels needed): distinct movies across all top-5 lists — TF-IDF 63, v1 45, v2 51; most repeated movie — TF-IDF 4, v1 14, v2 **14**. The hub correction widened the variety but did not remove the worst hub on the sample.

### Example 1 — success of meaning over words (M09)
*"A warm romance about two people finding each other unexpectedly."*

- **TF-IDF:** *Dangerous Love Affairs*, *Bhoot* (horror), *Justine and the Whip*, *Born to Kill*, *Salt and Fire* — all 0. The query becomes `warm romance people finding unexpectedly`; single lemmas match anywhere in 6,000 words of dialogue.
- **v1:** *Perfect Match* (2), *Extreme Dating* (1), *Dangerous Love Affairs* (0), *Girlfriend Boyfriend* (1), *Love, In Between* (0) → P@5 0.6.
- **v2:** *Quiet Wedding* (1), *Extreme Dating* (1), *Dangerous Love Affairs* (0), *High Society* (2), *The Uninvited* (1) → P@5 0.8.
- **Why:** the encoder compares the meaning of the whole request with the plot. But *Perfect Match* was found via a matched subtitle chunk that is a **love-song lyric** — the right film for a noisy reason.

### Example 2 — negation: fails for A and B, fixed in C (M23)
*"No horror, just tension and psychological mystery."*

- **TF-IDF:** stop-word removal turns it into `horror tension psychological mystery`, so "horror" becomes a wanted term → *Faces of Death II*, *The Town That Dreaded Sundown*, *Scared*… P@5 0.
- **v1:** keeps "No" but the encoder places "no horror" next to "horror" → *Scared*, *Ways to Live Forever*, *Bhoot*, *The Night Strangler*, *Crypt of the Vampire*. P@5 0.
- **v2:** the parser splits off `horror` as a negated term and penalises movies close to it → *The Uninvited*, *The Nameless*, *A Shadow of a Doubt*, *The Headless Woman* (2), *Des Pardes*. P@5 **0.8**.
- **Caveat:** v2 was designed *after* seeing this failure, so M23 is not an untouched test. The parser is rule based; in development it produced false negations (M06 "…a funny movie that does not take itself seriously" was parsed as *avoid "itself seriously"*, journal exp. 11) and was made conservative.

### Example 3 — a hub movie, unsolved by all semantic methods (M02)
*"I want a cozy, comforting movie for a rainy evening."*

- **v1:** rank 1 *Ways to Live Forever*, a drama about a boy dying of leukaemia (0). **v2:** *A Smile*, *Hum Kaun Hai* (horror), *Extreme Dating*, *Ways to Live Forever*, *The Uninvited* — **P@5 0**, v2's worst query.
- **Why:** this film's dialogue *talks about* horror films, fear, death and feelings ("you wanted horror films … we kept expecting monsters"), so some chunk is close to almost any mood query; it is in the top 5 of 14 of 27 queries and irrelevant in 10. Subtitle similarity measures what characters *talk about*, not how the film *feels*. Also, the sample may contain no truly cozy film: a 100-movie corpus limits every method.

### Example 4 — edge cases (M25–M28)
- *"gloooomy thriling mistree"* (M26): TF-IDF returns nothing (no word in vocabulary); v1 returns 1 relevant of 5 (human labels); v2 corrects spelling (status `ok_with_warning`) but "mistree" is still not turned into "mystery" (it becomes "mistress"; journal exp. 12); still 4 of 5 relevant because "gloomy thrilling" carries the request.
- *"I feel like watching something."* (M25): TF-IDF returns nothing (all words are filler), v1 returns five films anyway, v2 answers `low_confidence / too_vague`. A numeric similarity threshold could not separate vague queries from real ones (dev set: normal queries 0.542–0.663, gibberish "asdfgh qwerty zxcv" 0.571, "Surprise me." 0.628; journal exp. 14), so abstention is rule based.
- Empty query (M28): rejected by all three.

### Interpretation and alternative explanations
Meaning beats word overlap consistently (the TF-IDF gap holds on both label sets, and its CI excludes zero). Three alternative readings we cannot yet rule out: (a) **v2 vs v1** is within noise; (b) v2's negation gain may be partly **overfitting** to M19–M24, which motivated its design — the 50 dev queries were written for an unbiased check but are not labelled yet; (c) the AI-assisted labels may favour semantically plausible matches, i.e. the very thing embeddings produce.

## 3. Limits and Risks

- **Small, non-random sample.** 100 films, the first in file order; for M11 ("adventure full of wonder") and M15 ("forgotten seaside town") hardly any relevant film exists, so low scores partly measure the corpus, not the method. The full-corpus v2 run (78 queries) is unlabelled.
- **Labels.** The TF-IDF/v1 labels are one human judge, labelled in two sessions (blinding only partial). The v2 comparison uses a pooled sheet first labelled by an AI assistant: on the 394 pairs shared with the human labels, exact agreement is **78 %**, relevant/irrelevant agreement **85 %**, Cohen's κ **0.58** (moderate). Numbers across the two label sets are therefore not directly comparable, which is why we report both.
- **Subtitles are a weak mood signal** — our central hypothesis is only partly supported. Dialogue discusses emotions and genres rather than expressing atmosphere; it also creates hubs and carries noise (credits, lyrics).
- **Negation and spelling are rule based.** Not handled: "anything but…", sarcasm, double negation, real-word typos ("griping" for "gripping"), names. On the full index the typo version of "Cozy and a little bit grim" shares **0 of 10** films with the clean version. A false negation silently removes good films, which is worse for a user than an ignored one.
- **Data bias.** English-only archive, drama-heavy (50 %), many obscure titles, posters/plots from TMDB; popular films are not favoured, but users may not recognise any recommendation. No user history (cold-start by design): the system cannot personalise.
- **Untuned v2.** Fusion weights, λ and τ are placeholders; the BGE-base/BGE-small split was chosen by measured encoding time on a compute budget, not by accuracy.
- **Cost.** Full index: several GB, median **4.5 s per query** (p95 5.5 s, RTX 2060), dominated by scoring 2.25 M chunks — too slow for an interactive app without approximate search.

**What we would defend:** fixing the main comparison before labelling; pooled blind labelling; the corrected nDCG; separate dev and test queries. **What we would do differently / next:** a second human judge on a random subset of all three methods; label the dev set and tune the weights there, not on M01–M28; evaluate on a random sample from the full corpus; drop or down-weight the subtitle channel unless an ablation on the full corpus shows a gain; an approximate-nearest-neighbour index for latency.

*Full details: `docs/failure-journal.md` (14 dated experiments), `docs/semantic_v2_results.md` (generated metrics, intervals, per-query error tables), `docs/tfidf_pipeline.md`, `docs/semantic_pipeline.md`, `docs/semantic_v2.md`.*
