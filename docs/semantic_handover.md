# Semantic search – předávací dokumentace (pro Vojtu)

Stav k 20. 11. 2026 · autor: Tomáš · větev `semantic-search` (zatím není v `main`)

Tenhle dokument říká, **co je nového, co se smí a nesmí spouštět, co čím se přepíše** a jak pustit semantic search na plná data. Jak semantic search funguje uvnitř, popisuje `docs/semantic_pipeline.md`.

---

## 1. TL;DR

- **Plná data:** pusť v Colabu s GPU jen **trénink indexu** (`src.semantic train`). Je to dlouhý krok a dá se udělat hned, protože pozdější úpravy (oprava hubness) se týkají jen výpočtu skóre a nový trénink nebudou potřebovat.
- **Nepouštěj** na plných datech evaluační notebook `02_mood_query_evaluation.ipynb` ani `src.run_semantic_eval run` s výchozím nastavením. Obojí zapisuje do `evaluation/results/` a `evaluation/mood_judgments.csv`, a tím by přepsalo výsledky ze samplu, ke kterým tým právě píše známky.
- **`mood_judgments.csv`** má teď 394 řádků místo 119. Přidal jsem filmy, které doporučil semantic search. Hodnotit se má verze z větve `semantic-search`.
- Indexy z plných dat (`models/semantic_full/`, `models/tfidf_full/`) **nenahrávat na GitHub**: jsou velké a obsahují texty titulků.

---

## 2. Co je nového v repu

| Soubor | K čemu je |
|---|---|
| `src/semantic.py` | semantic search: `train` (vytvoří index) a `search` (dotaz → top filmy). Stejný styl jako `src/baseline.py`, načítá data jeho funkcí `load_dataset`. |
| `src/run_semantic_eval.py` | `run`: 28 dotazů × 4 varianty + doplnění tabulky hodnocení · `metrics`: srovnání TF-IDF vs. semantic z hotových známek |
| `notebooks/03_semantic_search_colab.ipynb` | běh v Google Colabu s GPU (sample i plná data) |
| `models/semantic_sample/` | hotový index pro sample (100 filmů). Texty titulků (`chunk_text.parquet`) nejsou commitnuté. |
| `evaluation/results/sem_*_mood_rankings.csv` | top 5 pro každý dotaz a variantu, se sloupcem `best_chunk` (úryvek titulků, který s dotazem seděl) |
| `evaluation/results/method_overlap.csv` | kolik filmů má každá varianta v top 5 společných s TF-IDF |
| `evaluation/results/semantic_query_status.csv`, `semantic_mood_results.json` | stav dotazů, plné výsledky |
| `evaluation/mood_judgments.csv` | **upravený**: +275 řádků (viz sekce 4) |
| `docs/semantic_pipeline.md` | jak semantic search funguje a proč |
| `docs/failure-journal.md` | **upravený**: Experiment 3–6 |
| `.gitignore` | **upravený**: ignoruje velké soubory semantic indexů |

Na tvoje soubory (`baseline.py`, `tfidf_text.py`, `preprocessing.py`, notebook `02_...`, `models/tfidf_sample/`) jsem **nesahal**.

---

## 3. Co se smí spouštět a co čím se přepíše

| Co | Kam to zapisuje | Smí se teď? |
|---|---|---|
| `python -m src.semantic train ... --output models/semantic_full` | jen `models/semantic_full/` | ✅ **ano**, to je ten krok pro plná data |
| `python -m src.semantic search ...` | nikam (jen vypíše výsledky) | ✅ ano, kdykoliv |
| `python -m src.baseline train ... --output models/tfidf_full` | jen `models/tfidf_full/` | ✅ ano |
| Notebook `02_mood_query_evaluation.ipynb` (jakýkoli index) | **přepíše** `evaluation/results/tfidf_*` a **přidá řádky** do `evaluation/mood_judgments.csv` | ⛔ **ne**, dokud se nedohodnotí sample |
| `python -m src.run_semantic_eval run` bez `--results-dir` | **přepíše** `evaluation/results/sem_*` a **přidá řádky** do `evaluation/mood_judgments.csv` | ⛔ **ne** |
| `python -m src.run_semantic_eval run ... --results-dir evaluation/results/full --judgments evaluation/mood_judgments_full.csv` | jen `evaluation/results/full/` a `evaluation/mood_judgments_full.csv` | ⚠️ technicky bezpečné, ale až po domluvě týmu (přidá další stovky řádků k hodnocení) |
| `python -m src.run_semantic_eval metrics` | `evaluation/results/comparison_*.csv` | ✅ ano, jen čte známky |
| Colab notebook, **Part A** (sample) | `models/semantic_sample/`, `evaluation/results/sem_*` | ❌ **nepouštět**: sample je hotový, rerun by soubory zbytečně přepsal |
| Colab notebook, **Part B** (full) | `models/semantic_timing/`, `models/semantic_full/` | ✅ ano. Poslední buňka s `run` je zakomentovaná. |

### Co je potřeba přepočítat, když se něco změní

| Změna | Nový `semantic train`? | Nový `run`? | Známky platí dál? |
|---|---|---|---|
| Oprava hubness / jiná agregace / jiná váha plot vs. titulky | ne | ano | ano pro filmy, které už ohodnocené jsou; nově doporučené filmy se dohodnotí |
| Nový / upravený dotaz v `mood_queries.json` | ne | ano | upravený dotaz → **nové ID** (jinak se použijí známky ke starému znění) |
| Změna dat (`clean_movies.parquet`, preprocessing, sloupec `text_embedding`) | **ano** | ano | ano pro stejné dvojice dotaz + film |
| Jiný embedding model nebo délka úseku (`--chunk-words`) | **ano** | ano | ano pro stejné dvojice dotaz + film |
| Doplněné známky v `mood_judgments.csv` | ne | ne | jen pustit `metrics` |

---

## 4. Tabulka hodnocení: co jsem přidal a jak se pozná metoda

**Co se změnilo:** `run` vzal top 5 filmů ze všech 4 semantic variant a každou dvojici **dotaz + film**, která v tabulce ještě nebyla, přidal jako nový řádek s prázdným `relevance`. Stávající řádky a známky zůstaly. Tabulka je seřazená podle `query_id`, takže nové řádky jsou zamíchané mezi původní, ne na konci. Nový sloupec nepřibyl.

**Proč tabulka neříká, která metoda film doporučila:** je to záměr, tzv. *pooling* a *slepé hodnocení*.
- Film doporučený oběma metodami je v tabulce **jen jednou**, takže se hodnotí jednou a známka platí pro obě metody. Kdyby byla tabulka zvlášť pro každou metodu, bylo by řádků ~680 a ke stejnému filmu by mohly vzniknout dvě různé známky.
- Hodnotitel neví, jestli film vybral TF-IDF, nebo semantic search, takže nemůže nevědomky nadržovat.

**Jak se zpětně pozná, co doporučila která metoda:** z výsledkových souborů. Každá metoda má svůj:

```
evaluation/results/tfidf_mood_rankings.csv          ← top 5 TF-IDF
evaluation/results/sem_meta_mood_rankings.csv       ← top 5 sem_meta
evaluation/results/sem_subs_max_mood_rankings.csv   ...
evaluation/results/sem_subs_top3_mood_rankings.csv
evaluation/results/sem_combined_mood_rankings.csv
```

`python -m src.run_semantic_eval metrics` vezme pro každou metodu její top 5 u každého dotazu a podle dvojice `query_id` + `movie_id` k nim dohledá známky v `mood_judgments.csv`. Spočítá Precision@5, MRR@5 a nDCG@5 (stejné vzorce jako notebook 02) a uloží je do `evaluation/results/comparison_metrics.csv`. Metody srovnává jen na dotazech, které mají ohodnocených všech 5 filmů u **každé** metody, aby byly průměry srovnatelné.

**Z toho plyne jediné pravidlo:** dokud se hodnotí, **výsledkové soubory se nesmí přepsat**. Jinak by známky patřily k jiným filmům, než jaké metoda doporučila. Proto jsou v sekci 3 ty zákazy. Před odevzdáním stav zamrazí git tag `prefinal`, který zadání stejně vyžaduje.

---

## 5. Plná data v Colabu (krok za krokem)

Proč Colab: na notebooku bez GPU trvalo zakódování samplu (4 155 úseků titulků) 152 s, tedy ~27 úseků/s. Plná data mají ~50 000 filmů × ~42 úseků ≈ 2 miliony úseků, což by na CPU trvalo ~21 hodin. Na GPU T4 odhadem desítky minut až hodinu.

1. Na GitHubu přepni na větev **`semantic-search`** → **Code → Download ZIP**, rozbal a nahraj složku na Google Drive (např. `MyDrive/Movie-recommendation-system/`).
2. Do `data/processed/` na Drive přidej **`clean_movies.parquet`**. Musí mít sloupce `imdb_id`, `tmdb_title`, `text_embedding` (volitelně `genres`, `release_year`, `clean_plot`).
3. Otevři v Colabu `notebooks/03_semantic_search_colab.ipynb`, **Runtime → Change runtime type → T4 GPU**.
4. Pusť první dvě buňky (připojení Drive, instalace). Zkontroluj cestu `REPO` a že se vypíše `GPU available: True`.
5. **Part A přeskoč** (sample je hotový).
6. **Part B:**
   - buňka s testem na 500 filmech vypíše počet filmů a čas. Odhad plného běhu = čas × (počet filmů / 500);
   - buňka s plným `train` vytvoří `models/semantic_full/`;
   - poslední buňka (`run`) je zakomentovaná, nech ji tak.
7. Pošli Tomášovi: čas z testu na 500 filmech a obsah `models/semantic_full/manifest.json` (počet filmů, úseků, čas). Patří to do reportu i journalu.

**Když něco selže:**

| Problém | Řešení |
|---|---|
| Colab spadne kvůli paměti (RAM) nebo se odpojí před koncem | přidej k `train` `--max-chunks-per-movie 20` (20 rovnoměrně vybraných úseků z filmu, ~2× rychlejší). Zapsat jako trade-off do journalu. |
| Pořád dochází paměť | přidej `--no-chunk-text` (neukládá texty úseků; vyhledávání funguje, jen bez úryvků `best_chunk`) |
| `Missing required columns` | parquet nemá `text_embedding` nebo `imdb_id`/`tmdb_title`: zkontrolovat výstup preprocessingu |
| Duplicitní `imdb_id` | semantic je zahodí s varováním (na rozdíl od baseline, která skončí chybou). Počet je v logu. |

Index z plných dat bude mít ~2–3 GB. **Nechat na Drive, necommitovat** (`.gitignore` to hlídá jen pro `.npy` a `chunk_text.parquet`, ostatní soubory jsou malé).

---

## 6. Co bude dál

1. Tým ohodnotí `mood_judgments.csv` (větev `semantic-search`), pak se pustí `metrics`.
2. Tomáš udělá opravu hubness (nemění index, jen výpočet skóre).
3. Pull request `semantic-search` → `main`.
4. Rozhodnutí, jestli evaluovat i na plných datech. Pokud ano, tak do oddělených souborů (`--results-dir evaluation/results/full --judgments evaluation/mood_judgments_full.csv`), a TF-IDF na plných datech pak také exportovat do `evaluation/results/full/`, ne do výchozí složky.
