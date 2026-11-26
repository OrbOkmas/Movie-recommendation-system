# Team Contributions

**Project:** Mood-Based Movie Recommendation · **Repository:** <https://github.com/OrbOkmas/Movie-recommendation-system>

| Member | What they did | Data | Comparison & errors | Limits & defence | Journal |
|---|---|:-:|:-:|:-:|:-:|
| **Vojta** | OPUS extraction, TMDB alignment and enrichment, cleaning and the two text representations (`src/preprocessing.py`); full-corpus EDA; 100-movie sample | ● | | ● | Exp. 1 |
| **Eliška** | TF-IDF baseline (`src/baseline.py`); the 28 shared test queries M01–M28; evaluation notebook and metric code reused by all methods; repository structure and document templates | | ● | ● | Exp. 2 |
| **Tomáš** | Semantic Search v1 (`src/semantic.py`): subtitle chunking, ablation of 4 variants, pooled blind labelling sheet, significance tests | | ● | ● | Exp. 3–5, 8–9 |
| **Matyáš** | Semantic Search v2 (`src/semantic_v2/`): field fusion, hub correction, negation/spelling/statuses, full-corpus index; dev query set, corrected nDCG, bootstrap intervals, generated results report; 70 tests | ● | ● | ● | Exp. 6–7, 10–14 |
| **Sam** | Human relevance labelling of all TF-IDF and v1 results (394 pairs) on one 0/1/2 scale; coordination and integration of the prefinal package (report, README) | | ● | ● | |
| **Matouš** | Streamlit application on top of v2 (`app.py`) | | | ● | |

Work was divided, but all members reviewed the report and can explain the team's main decisions: why meaning-based retrieval was compared with TF-IDF, why the main comparison was fixed before labelling, and why subtitles turned out to be a weak mood signal. Per-person details and file references: `docs/contributions_detailed.md`.
