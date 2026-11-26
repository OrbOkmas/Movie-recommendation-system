# Semantic Search v2 — design & experiment plan (pre-code)

Status: implemented except tuning, real-model runs and rerank (D10 = backup only) · branch: `semantic-v2` (from `semantic-search`) · owner: Matyáš

## 1. Goal and scope

Build a better *semantic* retriever than v1 (`../../src/semantic.py`: MiniLM-L6, one plot vector + subtitle chunks, fixed 0.5/0.5 fusion). TF-IDF stays the baseline. Show the improvement with an ablation ladder, and document where v2 still fails.

Fixed constraints
- No generative LLM anywhere (small encoder / cross-encoder models are allowed).
- English-only queries; too-long queries produce a warning.
- Pure `.py` modules with a CLI (same style as v1: `train`, `search`, `run`, `metrics`). Notebooks only for analysis.
- Never overwrite v1 results or the team's `mood_judgments.csv`; v2 writes to `../../evaluation/results/v2` and its own judgments file.
- Hardware: RTX 2060 laptop (6 GB, fp16). Possible second machine (RTX 5060) for the final run — not to be relied on.

## 2. What the data shows

Full corpus profile (`../../evaluation/results/v2/data_profile.json`, 51,616 movies, no duplicate ids, none unrankable):

| Fact | Value | Consequence |
|---|---|---|
| Missing fields | plot 0.5%, keywords 27.6%, genres 1.3%, dialogue 0.2% | Missing-field handling (D2) is needed mainly for keywords; the "no text at all" exclusion (D3) never triggers on this corpus but stays as a safety rule. |
| Plot length | median 43 words (5th pct 16, 95th pct 115) | Short texts: a stronger encoder is cheap here (3 x 51k short texts). |
| Keywords per movie | median 3, 95th pct 17, max 70 | Capped at 15 in the field text; long lists are noisy (places, "lost film", ...). |
| Dialogue | median 6,205 words per movie | about 2.25M chunks of 150 words (43.5 per movie) |
| Genres | Drama 50%, Comedy 29%, Romance 17%, Thriller 17%, Action 15%, Horror 11% | Skewed prior: drama-heavy corpus. |
| Years | 1895-2026, median 2003 | Spans the whole century including recent films. (An earlier statement that the corpus ends in 2016 came from the 100-movie sample and was wrong for the full corpus.) |

Data caveat: the `sample_clean_movies.csv` shared in chat was a lossy re-export (comma-separated, trailing empty columns, plots cut at the first comma). Its "13% of movies without plot" was an export artifact; the real rate is 0.5%.

## 3. Design decisions (with alternatives, for the defense)

| # | Decision | Why | Alternative not chosen |
|---|---|---|---|
| D1 | Separate vectors for `plot`, `keywords`, `genres`; title **not** embedded | v1's concatenated string lets short plots and title words dominate (journal Exp. 6: "Lie Detector" top-1 for 5 queries) | Single concatenated vector (v1) — kept only as the ablation baseline |
| D2 | Missing field ⇒ no vector, score undefined, weights renormalised over available fields | A zero/empty-string vector gives arbitrary similarity | Impute empty-string embedding |
| D3 | Movies with no plot, no keywords and no usable subtitles are not rankable | Nothing to match on | Keep them with title+genres only |
| D4 | Per-query z-score of each field/source score (corpus mean/std), then weighted sum | Raw cosines from different fields and models are on different scales | Raw weighted cosines (v1); reciprocal-rank fusion (kept as ablation) |
| D5 | Hub correction for the subtitle channel: subtract each movie's mean score to a background query bank (CSLS-style). The bank is built from templated mood phrases and random plot sentences, **never** from evaluation queries | Journal: one movie in 19/27 top-5 lists. Coordinate with Tomáš (his planned fix) so we do not build it twice | Per-query normalisation only (does not remove movie-level bias) |
| D6 | Negation via rule-based parser (spaCy dependencies + cue list: not, no, without, never, nothing, avoid, `-n't`); score = z(pos) − λ·max(0, z(neg) − τ) | Encoders place "no horror" next to "horror" (journal Exp. 6, point 3) | Keep "not" in the query (tried by v1, did not help); LLM parsing (out of scope) |
| D7 | Clause splitting only for enumerations of short adjective/noun phrases; never for sentences containing a verb. Final = α·whole-query + (1−α)·soft-min over clauses | "A funny romance where they dislike each other…" is one narrative; splitting it would hurt | Always split on "and"/"but" |
| D8 | Metadata-aware negation (ablation): negated term matched to genre/keyword vocabulary down-weights those movies | Cheap and likely strong for "no horror" / "no gore" | Only embedding-based penalty |
| D9 | Abstention: if top-1 fused score is below a threshold calibrated on dev edge cases, return status `low_confidence` | Dense retrieval always returns 5 movies, even for gibberish | Always return results |
| D10 | Cross-encoder rerank of top-50 on plot + keywords + genres (not dialogue). Kept only if it beats the no-rerank model on dev | Rerankers are trained on web search relevance; mood queries are out of domain | Always-on rerank |

## 4. Query handling contract

`search(query) -> {status, warnings, processed_query, results}`; it never raises on user input.

Pipeline: strip / collapse whitespace → empty check → length check → lowercase-insensitive normalisation (caps, repeated punctuation, letters repeated ≥3×) → spell correction (SymSpell, max edit distance 2, domain vocabulary from the corpus protected) → negation/clause parsing → encode.

| Status | When |
|---|---|
| `ok` | normal |
| `ok_with_warning` | > 60 words (configurable) → warning, query truncated to the encoder window; or a spelling correction was applied (returned as `processed_query`) |
| `low_confidence` | top-1 score under threshold (gibberish, "I feel like watching something") |
| `negation_only` | no positive clause ("Nothing too scary and nothing too sad") |
| `empty` | empty or whitespace |

Known expectation: edit distance 2 will probably **not** fix "mistree" → "mystery" (≈3 edits). This is a documented limitation, and M26 is in the locked test set, so we do not tune for it.

## 5. Scoring spec

For query q and movie m, with source set S = {plot, keywords, genres, subtitles}:
1. `s_f(q,m)` = cosine (subtitles: mean of top-3 chunk scores, then hub-corrected).
2. `z_f` = (s_f − μ_f(q)) / σ_f(q), statistics over all movies for that query.
3. `z(m) = Σ_f w_f z_f / Σ_f w_f` over fields available for m.
4. Query-level: `score = α·z_whole + (1−α)·softmin_T(z_clause_i) − λ·max(0, z_neg − τ)`.
5. Weights `w_f`, α, T, λ, τ tuned on **dev** queries with cross-validation over queries; test queries are not used.

## 6. Models and compute

| Class | Examples | Params | Max tokens (check `model.max_seq_length`) | Relative cost |
|---|---|---|---|---|
| v1 | all-MiniLM-L6-v2 | 22M | 256 | 1× |
| small-12L | bge-small-en-v1.5, e5-small-v2, gte-small, all-MiniLM-L12-v2 | 33M | 512 (bge/e5/gte); **128 for MiniLM-L12** | ~2× |
| base | bge-base-en-v1.5, e5-base-v2, gte-base, all-mpnet-base-v2 | 110M | 512 (mpnet 384) | ~6–8× |

- GPU memory is not the limit; **time on ~2.25M chunks is**. Measured on the RTX 2060 (500 movies = 21,748 chunks, `../../evaluation/results/v2/encoder_timing.csv`): minilm-l6 18.6 s, bge-small 32.0 s, bge-base 74.3 s, i.e. about 32 min, 55 min and 2.1 h for the full corpus (upper bounds: they include model loading).
- Allocation (main run A): **bge-base for the short fields** (~150k short texts, minutes); **small-12L or v1 model for the subtitle chunks** (2.25M texts: bge-small for the main run, bge-base and minilm-l6 as ablation runs). Whether the larger model actually helps mood retrieval is an empirical question, so it is an ablation, not an assumption.
- MiniLM-L12's 128-token window would force ~90-word chunks (more chunks) and cancel its benefit — a trap to document.
- Encoding must be **sharded and resumable** (manifest + shard files per movie range): the run can survive sleep/crash and be split with the second machine.
- Fallback decided in advance: `--max-chunks-per-movie 20` for the full run, recorded as a trade-off in the failure journal.
- Practical: close other GPU programs, disable laptop sleep, sort texts by length before batching, fp16.

## 7. Evaluation protocol

- **Dev set:** `../../evaluation/dev_queries.json` (50 queries; 33 `core` need human labels, 17 `robustness` are mostly label-free). **Test set:** M01–M28, locked; run once at the end.
- **Leakage disclosure:** M01–M28 failures are documented in the journal and motivated the design (negation, typos, hubs). Tuning uses dev only; the report states that the *design* was informed by the test-set failure categories. Optional strengthening: a teammate who has not seen the system writes ~15 fresh queries as a final holdout.
- **Labeled metrics:** P@5, MRR@5, nDCG@5 as in the team notebook, plus nDCG with the ideal ranking taken from the whole judged pool for the query, plus per-query bootstrap 95% CI.
- **Label-free metrics:** hub concentration (max appearances of one movie, distinct movies over all queries, Gini of appearances); typo stability (overlap@5 of D34/D35 vs D02/D23); paraphrase stability (P1, P2); status accuracy on edge cases; weak-genre hit rate (`weak_genre_expect`, a sanity check only, with known false negatives); latency per query; index size.
- **Label budget:** label only the final 3 configurations (v1 best, v2 final, one ablation), pooled and blind as the team does.
- **Ablation ladder (one row each):** v1 `sem_combined` → +encoder → +separate fields → +calibrated fusion → +hub correction → +negation → +clauses/soft-AND → +abstention → +rerank. Every negative result goes into the failure journal.

## 8. Repo layout

```
src/semantic_v2/
  config.py     dataclasses, model registry (prefixes, max tokens)
  fields.py     build field texts, missing-field masks
  encode.py     sharded, resumable encoding
  index.py      save / load index + manifest
  query.py      QueryProcessor (normalise, spell, length, negation, clauses)
  scoring.py    z-scores, fusion, hub correction, penalties, abstention
  rerank.py     optional cross-encoder
  cli.py        train | search | run | metrics
tests/          parser, chunk, status, scoring unit tests (dev_queries.json gold fields)
evaluation/dev_queries.json
evaluation/results/v2/
docs/semantic_v2.md  (+ failure-journal entries)
```

## 9. Risks and limitations to be able to defend

- Dialogue still "talks about" moods rather than carrying them; subtitles may add little (a negative result is still a result).
- 100-movie sample cannot show differences; conclusions need the larger run and labels.
- Corpus ends in 2016 and is English/OpenSubtitles-biased; obscure titles dominate.
- Rule-based negation fails on indirect phrasing ("anything but…"), sarcasm, double negation.
- Spelling correction can damage rare words and proper nouns.
- Hub correction depends on the background bank design.
- Human labels: small number of annotators, no agreement statistics yet.

## 10. Open items

1. Full dataset access (teammate) → run the profile script, confirm sample statistics.
2. Check with Tomáš whether his hubness fix exists on a branch.
3. Decide which spaCy model / SymSpell dictionary to pin in `requirements-semantic-v2.txt`.
4. Agree on 60-word warning threshold.
