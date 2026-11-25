
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

### Experiment 2 — TF-IDF Baseline: Query Processing and Retrieval Limitations

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

### Experiment 3 — 2026-11-20 — Semantic search: one embedding per movie is not possible

**Team member:** Tomáš

**What we tried:** The first idea was to encode the prepared `text_embedding` column (title, genres, keywords, plot and dialogue) into one vector per movie with a sentence-embedding model (`all-MiniLM-L6-v2`).

**Why we tried it:** It is the simplest set-up and mirrors the TF-IDF baseline (one vector per movie, cosine similarity).

**What went wrong:** Before implementing it we checked the document lengths in the 100-movie sample: subtitles have on average 6,204 words (median 6,163, max 16,788), the whole `text_embedding` field 33,000 characters on average. MiniLM reads at most 256 word-pieces (about 150–200 words) and silently truncates the rest. One vector per movie would therefore encode the title, keywords, plot and only the first ~100 words of dialogue — the subtitles, which are the main idea of our project, would be almost ignored.

**Evidence / Example:** *The Servant (1963)*: `text_embedding` starts with 77 words of metadata and plot, followed by 6,307 words of dialogue. Less than 2% of the dialogue would reach the model.

**What we changed:** We split each document into two parts: (1) metadata + plot → one vector per movie, (2) dialogue → windows of 150 words ("chunks", 41.6 chunks per movie on average in the sample), one vector per chunk. A movie's subtitle score is aggregated from its chunk scores (max, mean of the top 3, mean). Both parts are stored in one index, so plot-only, subtitles-only and combined ranking can be compared without re-encoding.

**What we learned:** Truncation was a problem for both methods (TF-IDF cuts at 4,000 words) but it is much more severe for transformers. How long documents are represented is a methodological decision we have to justify, not a technical detail.

### Experiment 4 — 2026-11-20 — Chunking: tiny trailing chunks and an off-by-one bug

**Team member:** Tomáš

**What we tried:** Fixed 150-word windows over the dialogue.

**Why we tried it:** Simple, fits the model's input window, no sentence segmentation needed (subtitle punctuation is unreliable).

**What went wrong:** (1) The last window of a movie can have only a few words. A 5-word chunk such as "No. No, I don't, sir." can get a high cosine similarity by chance, and with max-aggregation one such chunk would decide the movie's rank. (2) Our first fix (merge the short tail into the previous chunk) contained a bug: in `spans[-2][1] = spans.pop()[1]` Python evaluates `pop()` first, so the tail was merged into the wrong chunk (a 320-word text produced chunks of 320 + 150 words instead of 150 + 170).

**Evidence / Example:** Unit test on a 320-word text: expected chunk lengths `[150, 170]`, got `[320, 150]`.

**What we changed:** A tail shorter than a third of a window (< 50 words) is merged into the previous chunk; bug fixed and covered by the test. We also verified the vectorised top-k aggregation against a slow pandas `groupby().nlargest()` implementation (identical results).

**What we learned:** Test small helper functions on hand-made inputs before running them on hours of data.

### Experiment 5 — 2026-11-20 — Development environment without model access

**Team member:** Tomáš

**What we tried:** Develop and run the semantic pipeline in the same environment where the code was written.

**What went wrong:** The environment had no access to the Hugging Face model hub (download blocked, HTTP 403) and no GPU. Measured on a team laptop (CPU only): encoding the 100-movie sample (4,155 chunks) took 152 s, i.e. ~27 chunks/s. The full data (~50,000 movies × ~42 chunks ≈ 2 million chunks) would take roughly 21 hours on CPU.

**What we changed:** The pipeline was tested end-to-end with an offline stand-in encoder (`--model debug-hashing`, hashing bag-of-words — explicitly not semantic, never used for results). Real encoding runs in Google Colab with a T4 GPU (`notebooks/03_semantic_search_colab.ipynb`), with a 500-movie timing test first and an option `--max-chunks-per-movie` if the full run does not fit into a Colab session.

**Open issues found during testing (to be checked with the real model):**
- Dense retrieval **always returns results**, also for the vague query M25 ("I feel like watching something.") and the misspelled M26, where TF-IDF returned nothing. That is not automatically better: returning five random movies with high confidence may be worse than returning none. We plan to look at the score distribution and consider a minimum-similarity threshold.
- Four semantic variants add ~300 new query/movie pairs to `mood_judgments.csv` on the sample alone. We need to decide as a team whether to judge all variants or only the best two.

### Experiment 6 — 2026-11-20 — First real run of semantic search: one movie dominates, negation still fails

**Team member:** Tomáš

**What we tried:** `all-MiniLM-L6-v2` on the 100-movie sample, 27 non-empty mood queries, four variants: plot+metadata only (`sem_meta`), subtitles with max aggregation (`sem_subs_max`), subtitles with mean of 3 best chunks (`sem_subs_top3`), and 0.5/0.5 combination (`sem_combined`). No human relevance labels yet, so the observations below are qualitative.

**Why we tried it:** To test the project hypothesis that subtitles carry mood information that plot descriptions miss.

**What went wrong:**
1. **One movie dominates the subtitle variants ("hub" problem).** *Ways to Live Forever* (a drama about a boy with leukaemia) appears in the top 5 of 17/27 queries with `sem_subs_max` and 19/27 with `sem_subs_top3`, and is ranked first for 10 queries in `sem_subs_top3` — including "cozy, comforting movie for a rainy evening" (M02) and "a funny movie that is not childish" (M22). TF-IDF's most frequent movie appears only 4 times. The subtitle variants return only 40–43 distinct movies across all queries, TF-IDF 63.
2. **The matched chunks show why: dialogue that *talks about* moods and genres matches mood queries.** For "gloomy and thrilling" (M01) and "eerie and unsettling" (M03) the best chunk is: *"there's nothing terrifying about it. Look, you wanted horror films. This is the one. It was dead boring! We kept expecting monsters or demons…"*. The characters discuss horror films; the movie is not one. Subtitle similarity measures what people talk about, not how the film feels.
3. **Negation is not solved by embeddings either.** For "No horror, just tension and psychological mystery" (M23), 100% of `sem_meta` results are tagged Horror (60–80% in the other variants), although only 17% of the sample is horror. "Suspense without violence or gore" (M21) and "A dark mystery, but not something too scary" (M19): 60–80% horror. The sentence encoder puts "no horror" close to "horror". Keeping "not" in the query (which TF-IDF removes) did not help.
4. **Misspellings are not solved either.** "gloooomy thriling mistree" (M26) shares 0 movies with the correctly spelled "gloomy and thrilling" (M01) in three of the four variants; MiniLM splits unknown words into meaningless word pieces. The difference to TF-IDF is only that semantic search returns *something* instead of nothing.
5. **Plot-only (`sem_meta`) has its own failure:** short plots are matched on surface words. "Cozy, comforting movie for a rainy evening" (M02) ranks the kidnapping thriller *Pray* first; "Lie Detector" (a one-sentence plot about "hidden feelings") is ranked first for 5 different queries.

**Evidence / Example:** `evaluation/results/sem_*_mood_rankings.csv` (column `best_chunk`), `method_overlap.csv` (semantic and TF-IDF top-5 lists share only 12–17% of movies on average).

**What looks promising:** subtitles sometimes catch atmosphere that the plot cannot: for "a lonely city at midnight" (M14) the best chunk of *Love, In Between* is a song lyric about "lonely streets and a man standing there alone". Vague and misspelled queries get lower top-1 plot similarity (M25: 0.197, M26: 0.229, median of all queries 0.367), so a minimum-score threshold for "no confident match" may be feasible.

**What we changed / plan:** (1) human relevance labels for the pooled results before claiming which method is better; (2) test a hub correction — normalise each movie's subtitle scores (e.g. subtract the movie's average similarity to generic queries) so movies whose dialogue is "about everything" stop dominating; (3) report negation as an unsolved limitation of both approaches rather than as an advantage of embeddings.

### Experiment 7 — 2026-11-22 — Semantic v2: design numbers taken from a damaged copy of the sample

**Team member:** Matyáš

**What we tried:**

To plan v2 we profiled the 100-movie `sample_clean_movies.csv` (a copy with the transcript column removed and re-saved) to learn missing rates, plot lengths and the year range.

**Why we tried it:**

We wanted to base the design (missing-field handling, encoder for short texts, limitations) on facts about the data instead of guesses.

**What went wrong:**

The re-saved copy was damaged: it was comma-separated with about 315 empty trailing columns, plot text was cut at its first comma, and some rows were empty after `keywords`. Conclusions drawn from it were wrong: "13 % of movies have no plot", "plots have a median of 13 words", "the corpus ends in 2016".

**Evidence / Example:**

In the damaged copy the `plot_word_count` column said median 39 words while the actual `clean_plot` had median 13. The real profile of `clean_movies.parquet` (51,616 movies, `evaluation/results/v2/data_profile.json`): plot missing for 0.5 % (270 movies), keywords 27.6 %, plot median 43 words, years 1895–2026.

**What we changed:**

We added a streaming `profile` command and rebuilt every number in the plan from the full corpus. Statements that came from the damaged copy were removed from the plan and marked as corrected.

**What we learned:**

Never derive data facts from a re-exported file. Compare a derived value with its own metadata column (`plot_word_count` vs. the real text length) and profile the original data.

### Experiment 8 — 2026-11-23 — Semantic v2: the negation parser produced false negations

**Team member:** Matyáš

**What we tried:**

A rule-based parser that finds negation cues ("not", "without", "no", "never", "avoid", "n't"), cuts the negated phrase out of the query and penalises films similar to it, because the encoder treats "no horror" like "horror" (experiment 6).

**Why we tried it:**

Negation was the one failure of v1 that the encoder cannot fix by itself, and TF-IDF deletes "not" as a stop word.

**What went wrong:**

The first version also fired on sentences that are not negations. "Give me a funny, chaotic movie that does not take itself seriously." (M06) was parsed as "avoid *itself seriously*" because "take" was in the list of verbs that may follow "do not" ("can't take violence"). "We do not want anything too long either" kept the word "either" in the negated term, and removing a phrase left fragments such as "funny but, with ghosts".

**Evidence / Example:**

M06 produced a negated term although the user asks for something funny. The cases "I don't know what I want", "a genre I cannot name", "nothing but fun", "not only funny but moving" and "dislike each other" must not be parsed as negation.

**What we changed:**

We removed "take", added filler words to the span stop list, cleaned dangling conjunctions after removal, and wrote 7 gold-annotated negation dev queries (`evaluation/dev_queries.json`, tested in `tests/test_query.py`) plus explicit "must stay untouched" tests. Any case the parser is unsure about is left alone, because the whole query is always scored as well.

**What we learned:**

A false negation is worse than a missed one: it silently removes good films. Rules for natural language need negative tests (sentences that look like negation but are not), not only positive examples.

### Experiment 9 — 2026-11-23 — Semantic v2: spell correction invents words and cannot fix real-word errors

**Team member:** Matyáš

**What we tried:**

SymSpell correction (edit distance ≤ 2) with an English frequency dictionary plus the corpus vocabulary, to repair queries such as "gloooomy thriling mistree" (M26), which in v1 shared no movie with the correctly spelled query.

**Why we tried it:**

A subword tokenizer turns typos into unrelated tokens, so the encoder cannot recover them.

**What went wrong:**

The correction itself produced errors: "humor" was changed to "humour" (US spelling missing from the dictionary), "litle" became "title" instead of "little" (equal distance, higher frequency), "thriling" became "thriving". "mistree" (intended "mystery") is still wrong, and real words used as typos ("griping" for "gripping", "grimm" for "grim") are never corrected.

**Evidence / Example:**

On the full index the typo queries retrieve different films than the clean ones: D23 "Cozy and a little bit grim" vs. its typo version D35 shares 0 of 10 movies, D02 vs. D34 shares 4 of 10 (`evaluation/results/v2/v2_A_full_diag.txt`).

**What we changed:**

US/UK spelling variants are never "corrected", candidates must start with the same letter, missing, extra or transposed letters are preferred to substitutions, distance 2 is only allowed for non-substitution edits, and queries that are mostly unrecognisable are left unchanged (and flagged, see experiment 11). Every correction is shown to the user in a warning.

**What we learned:**

Edit-distance correction helps with simple typos but cannot fix real-word errors or names, and every correction is a risk. The honest result is a measured limit (typo overlap 0.0 and 0.4), not a claim that typos are solved.

### Experiment 10 — 2026-11-23 — Semantic v2: clause splitting cut phrases in the wrong places

**Team member:** Matyáš

**What we tried:**

Splitting a query at "and", "but", "with" and commas into clauses ("cozy" / "a little bit grim") and combining the clause scores with a soft minimum, so that a film must fit all moods.

**Why we tried it:**

One vector for "cozy and a little grim" averages two opposite moods.

**What went wrong:**

The first rule split topical phrases and adjective pairs: "A bleak drama about poverty and survival" became "bleak drama about poverty" + "survival", "…a beautiful but disturbing atmosphere" became "a beautiful" + "disturbing atmosphere" (M17), "the vibe of an old, forgotten seaside town" became "the vibe of an old" + "forgotten seaside town" (M15).

**Evidence / Example:**

The three queries above, run through the processor (see `tests/test_query.py`, which now fixes the expected clauses for the 6 multi-aspect dev queries and asserts that the 5 plot-like queries are not split).

**What we changed:**

Splitting is only allowed when the query contains no finite verb, no preposition other than "with", and no dangling "a + adjective" piece; at most 4 clauses of at most 5 words. Everything else is scored as one query. The whole query is always part of the final score (`alpha`).

**What we learned:**

A conservative splitter that does nothing is better than a clever one that damages the query; the benefit of splitting has to be shown by the ablation, not assumed.

### Experiment 11 — 2026-11-24 — Semantic v2: a numeric confidence threshold cannot detect vague or nonsense queries

**Team member:** Matyáš

**What we tried:**

Abstain ("no confident match") when the best raw similarity is below a threshold calibrated on dev queries (`calibrate`), so that "Surprise me" or gibberish does not return five confident-looking films (v1 returned five films for M25 "I feel like watching something").

**Why we tried it:**

It is the standard approach and needs no rules.

**What went wrong:**

With BGE the cosine similarities are compressed into a narrow band, and the confidence of nonsense queries lies inside the range of normal queries. The z-score of the best movie is also useless, because the top of a distribution is always several standard deviations high.

**Evidence / Example:**

Dev queries with the default settings (`evaluation/results/v2/v2_A_full_calibrate.txt`): normal queries 0.542–0.663, gibberish "asdfgh qwerty zxcv" 0.571, "Surprise me." 0.628. Best threshold: Youden J = 0.381, not separable. The top-1 z-score of the gibberish query (3.67) was above the median of the normal queries (3.22). Status accuracy of the full run before the fix: 12/17 (71 %), all 5 errors were vague/gibberish queries.

**What we changed:**

Rule-based abstention in the query processor (`too_vague` for queries made only of words without criteria, `unrecognized_words` for mostly unknown tokens); the numeric threshold stays off (`min_confidence = 0`). Unit tests cover 6 vague queries, gibberish, and short specific queries ("melancholy", "romantic comedy") that must still pass. The rerun of the diagnostics is in `evaluation/results/v2/v2_A_full_diag.txt` (the result before the fix is kept in `…_before_patch3.txt`).

**What we learned:**

Similarity scores of a dense encoder are not calibrated probabilities. This fix was motivated by test queries M25 and M27, so the sample results cannot count as an untouched test; this is stated in the results.

### Experiment 12 — 2026-11-23 — Semantic v2: the encoder was chosen

**Team member:** Matyáš

**What we tried:**

Replacing MiniLM-L6 by stronger encoders. We timed `minilm-l6`, `bge-small` and `bge-base` as the chunk encoder on 500 movies (21,748 chunks) on an RTX 2060 laptop.

**Why we tried it:**

2.25 M subtitle chunks have to be encoded; the time decides what is possible before the deadline.

**What went wrong:**

Nothing failed, but we could not decide by quality: without labels we cannot show that a bigger encoder is better for mood retrieval, and the assumption "bge-base cannot be worse than bge-small" is untested (published gains are small and measured on question-answering benchmarks, and subtitles are our weakest signal).

**Evidence / Example:**

18.6 s, 32.0 s and 74.3 s per 500 movies (`evaluation/results/v2/encoder_timing.csv`), i.e. about 32 min, 55 min and 2.1 h for 51,616 movies. The main index (bge-base for plot/keywords/genres, bge-small for chunks) took 3,474 s (26 shards, 2,254,879 chunks).

**What we changed:**

The expensive channel got the cheaper encoder, the short fields got the stronger one, and the other two encoder combinations are built as ablation indexes (`scripts/build_full_indexes.ps1`). The choice is documented as a decision under a time budget.

**What we learned:**

Report what was measured (time) separately from what was assumed (quality). The ablation must test the assumption before the final report claims anything.

### Experiment 13 — 2026-11-22 — The shared nDCG@5 cannot see relevant movies that were not retrieved

**Team member:** Matyáš

**What we tried:**

Reusing the team's metric definition (`notebooks/02_mood_query_evaluation.ipynb`, `run_semantic_eval.py`) to compare v2 with TF-IDF and semantic v1.

**Why we tried it:**

All methods should be measured identically on the same judgments.

**What went wrong:**

The ideal DCG in that definition is computed from the same five retrieved movies, so a list of five "somewhat relevant" movies gets nDCG@5 = 1.0 even if the pool contains clearly relevant movies it missed. The metric rewards ordering, not retrieval quality.

**Evidence / Example:**

Unit test `tests/test_evaluate.py`: a list of five movies with relevance 1 while the judged pool also holds two movies with relevance 2 gets 1.00 with the team definition and about 0.47 when the ideal ranking comes from the whole judged pool.

**What we changed:**

We report both: `ndcg@k_team` (identical to the team notebook, for comparability) and `ndcg@k_pool` (ideal from all judged movies of the query), with per-query bootstrap intervals. We also report P@5 and MRR@5 unchanged.

**What we learned:**

A metric inherited from a teammate still needs a sanity test. The pool-based version is also imperfect (the pool depends on which methods were pooled), which is why both are shown.