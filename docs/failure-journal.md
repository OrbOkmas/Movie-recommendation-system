# Failure Journal: Mood-based Movie Recommender

Data: OPUS OpenSubtitles + TMDB (51,616 movies). Unless stated otherwise, experiments run on a 100-movie sample with 28 mood queries (M01–M28). Tags mark the three recurring problems: **N** negation, **S** misspellings, **H** hub movies that dominate results.

### 1 · 2026-11-18 · Data pipeline: missing IDs, Windows locks and memory errors · Vojta
**Tried:** stream the OPUS OpenSubtitles v2018 English corpus (446,612 XML files), match IMDb IDs to TMDB, lemmatise with multi-core spaCy on a Windows laptop, then run pandas EDA on the full data.
**Failed:** `opustools` pulled multi-GB bilingual alignment files, and the XML `<meta>` headers were stripped, giving **0 of 446,612** IMDb IDs. Over 63% of IDs are TV episodes; only matched movies were saved, so every restart re-queried them against TMDB. 11 spaCy workers crashed on a venv file lock (`OSError [Errno 22]`), and the neural pipeline stalled on >30,000-word files. `.explode()` on the full DataFrame copied the subtitle text and failed with `ArrowMemoryError` (3.2 GB).
**Then:** parse the year and IMDb ID from archive paths (140,044 IDs, 51,616 films), log checked IDs, use single-process spaCy + WordNet lemmatiser, cap documents at 10,000 words, fall back to the plot when subtitles have <50 words, store as Parquet and slice metadata columns before EDA.

### 2 · 2026-11-19 · TF-IDF baseline loses negation, typos and vague queries (N, S) · Eliška
**Tried:** TF-IDF over metadata, plot and subtitles, with filler and stop-word removal and lemmatisation, as an interpretable lexical baseline.
**Failed:** M25 "I feel like watching something", M26 "gloooomy thriling mistree" and empty M28 return **no results**. Removing "not" turns "dark but not scary" into "dark scary". M01 "gloomy and thrilling" returned comedies. 7 of 100 documents were cut at the 4,000-word limit.
**Then:** kept these edge cases in the query set so later methods face them; moved to semantic search. P@5 0.322, MRR@5 0.456, nDCG@5 0.481.

### 3 · 2026-11-20 · One embedding per movie would ignore the subtitles · Tomáš
**Tried:** one `all-MiniLM-L6-v2` vector per movie, mirroring TF-IDF.
**Failed (checked first):** subtitles average 6,204 words; MiniLM reads ~150–200 and silently truncates. For *The Servant* under 2% of the dialogue would reach the model.
**Then:** one plot + metadata vector plus 150-word subtitle chunks (41.6 per movie), aggregated by max / top-3 mean / mean.

### 4 · 2026-11-20 · Chunking: tiny tails and an evaluation-order bug · Tomáš
**Failed:** a 5-word last chunk ("No. No, I don't, sir.") can win a max-aggregated ranking by chance. Our fix `spans[-2][1] = spans.pop()[1]` merged the tail into the wrong chunk, because `pop()` runs first: `[320, 150]` instead of `[150, 170]`.
**Then:** tails under 50 words merge into the previous chunk, with a unit test. **Lesson:** test helpers on hand-made inputs before hours of encoding.

### 5 · 2026-11-20 · First semantic run: one film everywhere, negation and typos unsolved (H, N, S) · Tomáš
**Tried:** four variants (plot only, subtitles max, subtitles top-3, combined) to test whether dialogue carries mood.
**Failed:** **(H)** *Ways to Live Forever* (a leukaemia drama) is in the top 5 for 19 of 27 queries and first for 10, including "cozy, comforting movie for a rainy evening". Its matched chunk *talks about* horror films ("you wanted horror films… we kept expecting monsters"), so similarity measures what characters discuss, not how a film feels. **(N)** For M23 "No horror, just tension…" 100% of plot-only results are horror (17% of the sample). Keeping "not" did not help. **(S)** M26 shares 0 movies with M01 in 3 of 4 variants. Dense retrieval now always returns 5 films, even for M25.
**Then:** label before claiming; plan a hub correction; report negation as unsolved by both methods.

### 6 · 2026-11-22 · v2 design numbers came from a damaged file · Matyáš
**Tried:** profiling a re-saved copy of the sample to plan v2.
**Failed:** the copy had ~315 empty columns and plots cut at the first comma, giving wrong facts ("13% have no plot", "median plot 13 words", "corpus ends in 2016").
**Then:** a streaming `profile` of the full parquet: plot missing 0.5%, median 43 words, years 1895–2026.

### 7 · 2026-11-22 · The team's nDCG@5 cannot see missed relevant films · Matyáš
**Failed:** the ideal DCG uses only the five retrieved films, so five "somewhat relevant" films score **1.00**. Against the judged pool, the same list scores ~0.47 (unit test).
**Then:** report both `ndcg@k_team` (comparable) and `ndcg@k_pool`, with bootstrap intervals. **Lesson:** a shared metric still needs a sanity test.

### 8 · 2026-11-23 · Human labels: semantic wins, but subtitles help less than hoped (H, N) · Tomáš
**Tried:** pooled, blind 0/1/2 labels for 275 new pairs plus 119 TF-IDF ones. The main comparison (TF-IDF vs combined) was fixed *before* labelling.
**Found:** combined P@5 **0.461** vs 0.322, nDCG@5 0.631 vs 0.481, better on 14 of 23 queries (P@5 +0.14, 95% CI 0.03–0.25, Wilcoxon p = 0.045). But **plot alone (nDCG 0.657) beats subtitles alone (0.587–0.590)**: our core hypothesis is only partly supported. **(H)** *Ways to Live Forever* is in 14 of 27 lists, graded 0 in 10. **(N)** M23 has 0 relevant films for every method; its best "match" was a translator credit, which cleaning does not remove. **Caveat:** one judge, partial blinding.

### 9 · 2026-11-23 · Spelling correction v1: tested, not adopted (S) · Tomáš
**Failed:** a general dictionary breaks film words ("horor" → "honor", "noir" → "nor"). Adding plot vocabulary gave "thriling" → "thriving", "mistree" → "mistress". A hand-made mood list worked, but was written while looking at the test queries.
**Then:** not adopted (1 of 28 queries has typos). **Lesson:** a general tool can be worse than nothing in a narrow domain.

### 10 · 2026-11-23 · v2 encoder chosen by time, not quality · Matyáš
**Tried:** timing MiniLM, bge-small and bge-base on 500 movies: 18.6 s, 32.0 s, 74.3 s (32 min to 2.1 h for the full corpus).
**Failed:** without labels we cannot show a bigger encoder is better for mood.
**Then:** bge-base for short fields, bge-small for the 2.25 M chunks (3,474 s); the other combinations are built as ablations.

### 11 · 2026-11-23 · v2 negation parser invented negations (N) · Matyáš
**Tried:** a rule-based parser that cuts negated phrases out and penalises films similar to them.
**Failed:** M06 "a funny movie that does **not take itself seriously**" became "avoid *itself seriously*"; "anything too long either" kept "either"; removal left fragments like "funny but, with ghosts".
**Then:** fixed the rules, added 7 gold negation dev queries and "must stay untouched" tests ("I don't know what I want", "nothing but fun"). **Lesson:** a false negation is worse than a missed one, because it silently removes good films.

### 12 · 2026-11-23 · v2 spell correction still fails (S) · Matyáš
**Tried:** SymSpell (edit distance ≤ 2) with an English dictionary plus corpus vocabulary.
**Failed:** "humor" → "humour", "litle" → "title", "thriling" → "thriving"; "mistree" stays wrong; real-word typos ("griping", "grimm") are never caught. On the full index D23 vs its typo version shares **0 of 10** films, D02 vs D34 **4 of 10**.
**Then:** stricter candidate rules, no US/UK changes, every correction shown to the user. Typos are a measured limit, not solved.

### 13 · 2026-11-23 · v2 clause splitting cut phrases apart · Matyáš
**Failed:** splitting at "and / but / with / ," broke "poverty and survival" and M17 "a beautiful but disturbing atmosphere" into "a beautiful" + "disturbing atmosphere".
**Then:** split only short verb-free queries (≤ 4 clauses of ≤ 5 words) and always score the whole query too. Whether splitting helps is left to the ablation.

### 14 · 2026-11-24 · A confidence threshold cannot detect vague or nonsense queries · Matyáš
**Tried:** abstain when the best similarity is below a threshold calibrated on dev queries.
**Failed:** BGE similarities are compressed: normal queries 0.542–0.663, gibberish "asdfgh qwerty zxcv" 0.571, "Surprise me." 0.628 (Youden J 0.381, not separable). Status accuracy 12 of 17, all 5 errors vague or gibberish.
**Then:** rule-based `too_vague` and `unrecognized_words` statuses; threshold off. The fix was motivated by M25 and M27, so the sample is no longer an untouched test set.

**Still open:** negation by degree ("not too scary"), real-word typos, the hub problem in the subtitle channel, and a second judge. v2 has no human labels yet.
