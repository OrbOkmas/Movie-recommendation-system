# Failure Journal: Mood-based Movie Recommender

Data: OPUS OpenSubtitles + TMDB (51,616 movies). Unless stated otherwise, experiments run on a 100-movie sample with 28 mood queries (M01–M28). Tags mark the three recurring problems: **N** negation, **S** misspellings, **H** hub movies that dominate results.

### Experiment 1 · 2026-11-18 · Data pipeline: missing IDs, Windows locks and memory errors
**Team member:** Vojta

**What we tried:** stream the OPUS OpenSubtitles v2018 English corpus (446,612 XML files), match IMDb IDs to TMDB, lemmatise with multi-core spaCy on a Windows laptop, then run pandas EDA on the full data.

**What went wrong:**

- `opustools` pulled multi-GB bilingual alignment files, and the XML `<meta>` headers were stripped: **0 of 446,612** IMDb IDs found.
- Over 63% of IDs are TV episodes. Only matched movies were saved, so every restart re-queried them against TMDB.
- 11 spaCy workers crashed on a venv file lock (`OSError [Errno 22]`); the neural pipeline stalled on >30,000-word files.
- `.explode()` on the full DataFrame copied the subtitle text and failed with `ArrowMemoryError` (3.2 GB).

**What we changed:**

- Parse year and IMDb ID from archive paths (140,044 IDs, 51,616 films) and log checked IDs.
- Single-process spaCy + WordNet lemmatiser; documents capped at 10,000 words; plot fallback when subtitles have <50 words.
- Store as Parquet and slice metadata columns before EDA.

### Experiment 2 · 2026-11-19 · TF-IDF baseline loses negation, typos and vague queries (N, S)
**Team member:** Eliška

**What we tried:** TF-IDF over metadata, plot and subtitles, with filler and stop-word removal and lemmatisation, as an interpretable lexical baseline.

**What went wrong:**

- M25 "I feel like watching something", M26 "gloooomy thriling mistree" and empty M28 return **no results**.
- Removing "not" turns "dark but not scary" into "dark scary".
- M01 "gloomy and thrilling" returned comedies.
- 7 of 100 documents were cut at the 4,000-word limit.

**What we changed:** kept these edge cases in the query set so later methods face them; moved to semantic search.

**Result:** P@5 0.322, MRR@5 0.456, nDCG@5 0.481.

### Experiment 3 · 2026-11-20 · One embedding per movie would ignore the subtitles
**Team member:** Tomáš

**What we tried:** one `all-MiniLM-L6-v2` vector per movie, mirroring TF-IDF.

**What went wrong (checked before building):**

- Subtitles average 6,204 words; MiniLM reads ~150–200 and silently truncates.
- For *The Servant*, under 2% of the dialogue would reach the model.

**What we changed:** one plot + metadata vector plus 150-word subtitle chunks (41.6 per movie), aggregated by max / top-3 mean / mean.

### Experiment 4 · 2026-11-20 · Chunking: tiny tails and an evaluation-order bug
**Team member:** Tomáš

**What went wrong:**

- A 5-word last chunk ("No. No, I don't, sir.") can win a max-aggregated ranking by chance.
- Our fix `spans[-2][1] = spans.pop()[1]` merged the tail into the wrong chunk, because `pop()` runs first: `[320, 150]` instead of `[150, 170]`.

**What we changed:** tails under 50 words merge into the previous chunk, covered by a unit test.

**What we learned:** test helpers on hand-made inputs before hours of encoding.

### Experiment 5 · 2026-11-20 · First semantic run: one film everywhere, negation and typos unsolved (H, N, S)
**Team member:** Tomáš

**What we tried:** four variants (plot only, subtitles max, subtitles top-3, combined) to test whether dialogue carries mood.

**What went wrong:**

- **(H)** *Ways to Live Forever* (a leukaemia drama) is in the top 5 for 19 of 27 queries and first for 10, including "cozy, comforting movie for a rainy evening".
- **(H)** Its matched chunk *talks about* horror films ("you wanted horror films… we kept expecting monsters"): similarity measures what characters discuss, not how a film feels.
- **(N)** For M23 "No horror, just tension…", 100% of plot-only results are horror (17% of the sample). Keeping "not" did not help.
- **(S)** M26 shares 0 movies with M01 in 3 of 4 variants.
- Dense retrieval always returns 5 films, even for M25.

**What we changed:** label before claiming; plan a hub correction; report negation as unsolved by both methods.

### Experiment 6 · 2026-11-22 · v2 design numbers came from a damaged file
**Team member:** Matyáš

**What we tried:** profiling a re-saved copy of the sample to plan v2.

**What went wrong:**

- The copy had ~315 empty columns and plots cut at the first comma.
- This gave wrong facts: "13% have no plot", "median plot 13 words", "corpus ends in 2016".

**What we changed:** a streaming `profile` of the full parquet: plot missing 0.5%, median 43 words, years 1895–2026.

### Experiment 7 · 2026-11-22 · The team's nDCG@5 cannot see missed relevant films
**Team member:** Matyáš

**What went wrong:**

- The ideal DCG uses only the five retrieved films, so five "somewhat relevant" films score **1.00**.
- Against the judged pool, the same list scores ~0.47 (unit test).

**What we changed:** report both `ndcg@k_team` (comparable) and `ndcg@k_pool`, with bootstrap intervals.

**What we learned:** a shared metric still needs a sanity test.

### Experiment 8 · 2026-11-23 · Human labels: semantic wins, but subtitles help less than hoped (H, N)
**Team member:** Tomáš

**What we tried:** pooled, blind 0/1/2 labels for 275 new pairs plus 119 TF-IDF ones. The main comparison (TF-IDF vs combined) was fixed *before* labelling.

**What we found:**

- Combined P@5 **0.461** vs 0.322, nDCG@5 0.631 vs 0.481; better on 14 of 23 queries (P@5 +0.14, 95% CI 0.03–0.25, Wilcoxon p = 0.045).
- **Plot alone (nDCG 0.657) beats subtitles alone (0.587–0.590)**: our core hypothesis is only partly supported.
- **(H)** *Ways to Live Forever* is in 14 of 27 lists, graded 0 in 10.
- **(N)** M23 has 0 relevant films for every method; its best "match" was a translator credit, which cleaning does not remove.

**Caveat:** one judge, partial blinding.

### Experiment 9 · 2026-11-23 · Spelling correction v1: tested, not adopted (S)
**Team member:** Tomáš

**What went wrong:**

- A general dictionary breaks film words: "horor" → "honor", "noir" → "nor".
- Adding plot vocabulary gave "thriling" → "thriving", "mistree" → "mistress".
- A hand-made mood list worked, but was written while looking at the test queries.

**What we changed:** not adopted (1 of 28 queries has typos).

**What we learned:** a general tool can be worse than nothing in a narrow domain.

### Experiment 10 · 2026-11-23 · v2 encoder chosen by time, not quality
**Team member:** Matyáš

**What we tried:** timing MiniLM, bge-small and bge-base on 500 movies: 18.6 s, 32.0 s, 74.3 s (32 min to 2.1 h for the full corpus).

**What went wrong:** without labels we cannot show a bigger encoder is better for mood.

**What we changed:** bge-base for short fields, bge-small for the 2.25 M chunks (3,474 s); the other combinations are built as ablations.

### Experiment 11 · 2026-11-23 · v2 negation parser invented negations (N)
**Team member:** Matyáš

**What we tried:** a rule-based parser that cuts negated phrases out and penalises films similar to them.

**What went wrong:**

- M06 "a funny movie that does **not take itself seriously**" became "avoid *itself seriously*".
- "anything too long either" kept "either".
- Removal left fragments like "funny but, with ghosts".

**What we changed:** fixed the rules; added 7 gold negation dev queries and "must stay untouched" tests ("I don't know what I want", "nothing but fun").

**What we learned:** a false negation is worse than a missed one, because it silently removes good films.

### Experiment 12 · 2026-11-23 · v2 spell correction still fails (S)
**Team member:** Matyáš

**What we tried:** SymSpell (edit distance ≤ 2) with an English dictionary plus corpus vocabulary.

**What went wrong:**

- "humor" → "humour", "litle" → "title", "thriling" → "thriving"; "mistree" stays wrong.
- Real-word typos ("griping", "grimm") are never caught.
- On the full index, D23 vs its typo version shares **0 of 10** films; D02 vs D34 shares **4 of 10**.

**What we changed:** stricter candidate rules, no US/UK changes, every correction shown to the user.

**What we learned:** typos are a measured limit, not solved.

### Experiment 13 · 2026-11-23 · v2 clause splitting cut phrases apart
**Team member:** Matyáš

**What went wrong:**

- Splitting at "and / but / with / ," broke "poverty and survival".
- M17 "a beautiful but disturbing atmosphere" became "a beautiful" + "disturbing atmosphere".

**What we changed:** split only short verb-free queries (≤ 4 clauses of ≤ 5 words), and always score the whole query too.

**What we learned:** whether splitting helps is left to the ablation.

### Experiment 14 · 2026-11-24 · A confidence threshold cannot detect vague or nonsense queries
**Team member:** Matyáš

**What we tried:** abstain when the best similarity is below a threshold calibrated on dev queries.

**What went wrong:**

- BGE similarities are compressed: normal queries 0.542–0.663, gibberish "asdfgh qwerty zxcv" 0.571, "Surprise me." 0.628 (Youden J 0.381, not separable).
- Status accuracy was 12 of 17; all 5 errors were vague or gibberish.

**What we changed:** rule-based `too_vague` and `unrecognized_words` statuses; threshold off.

**Caveat:** the fix was motivated by M25 and M27, so the sample is no longer an untouched test set.
