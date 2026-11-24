"""Run the shared mood queries through semantic-search variants and compare them with TF-IDF.

Two sub-commands (run from the repository root):

    # 1) retrieve: write rankings for every variant (does not touch any judgment file)
    python -m src.run_semantic_eval run --index models/semantic_sample

    # 2) metrics from evaluation/mood_judgment_tfidf.csv + evaluation/mood_judgment_semantic.csv
    python -m src.run_semantic_eval metrics

The main, interactive evaluation is notebooks/04_semantic_mood_evaluation.ipynb (same metric code); it also
creates the labeling sheet mood_judgment_semantic.csv. This CLI is the scripted equivalent.

    # full data: keep everything separate from the judged sample results; --judgments also adds the new
    # unjudged pairs to that (separate) labeling sheet
    python -m src.run_semantic_eval run --index models/semantic_full --reference data/processed/clean_movies.parquet \
        --results-dir evaluation/results/full --judgments evaluation/mood_judgments_full.csv

Variants share one index (no re-encoding); they differ only in how scores are computed:
    sem_meta            metadata + plot vector only
    sem_subs_max        subtitles, movie score = best chunk
    sem_subs_top3       subtitles, movie score = mean of 3 best chunks
    sem_combined        0.5 * meta + 0.5 * subtitles(top3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.semantic import SemanticMovieRecommender, SemanticError
from src.metrics import metrics_at_k

ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "evaluation" / "mood_queries.json"
RESULTS = ROOT / "evaluation" / "results"
SEMANTIC_DIR = RESULTS / "semantic1"
COMPARISON_DIR = RESULTS / "comparison"
TFIDF_DIR = RESULTS / "tfidf"

TFIDF_JUDGMENTS = ROOT / "evaluation" / "mood_judgment_tfidf.csv"      # human labels for TF-IDF (read only)
JUDGMENTS = ROOT / "evaluation" / "mood_judgment_semantic.csv"         # human labels for semantic-only pairs
REFERENCE = ROOT / "data" / "processed" / "sample_clean_movies.csv"

def _find_tfidf_rankings() -> Path:
    p = TFIDF_DIR / "tfidf_mood_rankings.csv"
    if not p.is_file() and (RESULTS / "tfidf_mood_rankings.csv").is_file():
        return RESULTS / "tfidf_mood_rankings.csv"
    return p

TFIDF_RANKINGS = _find_tfidf_rankings()

VARIANTS = {
    "sem_meta":      dict(source="meta"),
    "sem_subs_max":  dict(source="subtitles", aggregation="max"),
    "sem_subs_top3": dict(source="subtitles", aggregation="topk_mean", agg_k=3),
    "sem_combined":  dict(source="combined", aggregation="topk_mean", agg_k=3, meta_weight=0.5),
}
LABEL_COLUMNS = ["query_id", "movie_id", "query", "title", "genres", "plot_snippet", "relevance", "reviewer", "notes"]


# --------------------------------------------------------------------------- retrieval

def run(index: Path, top_k: int, variants: list[str], pool_variants: list[str], reference: Path,
        extend_sheet: bool = False) -> None:
    model = SemanticMovieRecommender.load(index)
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, status_rows, full = [], [], []
    for name in variants:
        params = VARIANTS[name]
        for item in queries:
            rec = {"query_id": item["query_id"], "category": item.get("category", ""),
                   "query": item["query"], "test_purpose": item.get("test_purpose", ""),
                   "method": name, "params": params, "model": model.config.model_name, "recommendations": []}
            try:
                rec["recommendations"] = model.search(item["query"], top_k=top_k, **params)
                rec["status"] = "ok" if rec["recommendations"] else "no_matches"
            except (SemanticError, ValueError, TypeError) as exc:
                rec["status"], rec["error"] = "error", str(exc)
            full.append(rec)
            status_rows.append({k: rec.get(k, "") for k in ("method", "query_id", "category", "query", "status", "error")}
                               | {"returned": len(rec["recommendations"])})
            for m in rec["recommendations"]:
                all_rows.append({"method": name, "query_id": rec["query_id"], "category": rec["category"],
                                 "query": rec["query"], **{k: m.get(k, "") for k in (
                                     "rank", "movie_id", "title", "score", "meta_score", "subtitle_score",
                                     "genres", "release_year", "best_chunk")}})

    rankings = pd.DataFrame(all_rows)
    for name in variants:
        rankings[rankings.method == name].to_csv(SEMANTIC_DIR / f"{name}_mood_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(status_rows).to_csv(SEMANTIC_DIR / "semantic_query_status.csv", index=False, encoding="utf-8-sig")
    (SEMANTIC_DIR / "semantic_mood_results.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved rankings for {len(variants)} variants to {SEMANTIC_DIR.relative_to(ROOT)}")

    overlap = overlap_table(rankings, top_k)
    if overlap is not None:
        overlap.to_csv(SEMANTIC_DIR / "method_overlap.csv", index=False, encoding="utf-8-sig")
        print("\nMean overlap of top-k lists with TF-IDF (share of shared movies):")
        print(overlap.drop(columns="query_id").mean(numeric_only=True).round(3).to_string())

    if extend_sheet:
        added = extend_judgments(rankings[rankings.method.isin(pool_variants)], queries, reference)
        print(f"\nAdded {added} new query/movie pairs to {JUDGMENTS} (existing labels kept).")
    else:
        print("\nJudgment files not touched (pass --judgments <file> to write a labeling sheet).")


def overlap_table(rankings: pd.DataFrame, top_k: int) -> pd.DataFrame | None:
    """|top-k(semantic) ∩ top-k(tfidf)| / k per query — computable before any human labels."""
    tfidf_file = _find_tfidf_rankings()
    if not tfidf_file.is_file():
        print("No TF-IDF rankings found; skipping overlap.")
        return None
    tf = pd.read_csv(tfidf_file, dtype={"movie_id": str})
    tf_sets = tf.groupby("query_id")["movie_id"].apply(set)
    rows = []
    for qid in sorted(set(rankings.query_id) | set(tf_sets.index)):
        row = {"query_id": qid}
        for name, grp in rankings.groupby("method"):
            s = set(grp.loc[grp.query_id == qid, "movie_id"].astype(str))
            row[f"{name}_vs_tfidf"] = len(s & tf_sets.get(qid, set())) / top_k
        rows.append(row)
    return pd.DataFrame(rows)


def extend_judgments(rankings: pd.DataFrame, queries: list[dict], reference: Path) -> int:
    plot = {}
    if reference.is_file():
        if reference.suffix.lower() in {".parquet", ".pq"}:
            ref = pd.read_parquet(reference, columns=["imdb_id", "clean_plot"]).astype(str)
        else:
            ref = pd.read_csv(reference, sep=";", encoding="utf-8-sig", usecols=["imdb_id", "clean_plot"], dtype=str)
        plot = dict(zip(ref.imdb_id, ref.clean_plot.fillna("")))
    qtext = {q["query_id"]: q["query"] for q in queries}
    old = (pd.read_csv(JUDGMENTS, encoding="utf-8-sig", dtype={"movie_id": str})
           if JUDGMENTS.is_file() else pd.DataFrame(columns=LABEL_COLUMNS))
    seen = set(zip(old.query_id.astype(str), old.movie_id.astype(str)))
    new = (rankings.drop_duplicates(["query_id", "movie_id"])
           .loc[lambda d: [(q, str(m)) not in seen for q, m in zip(d.query_id, d.movie_id)]])
    new = pd.DataFrame({"query_id": new.query_id, "movie_id": new.movie_id.astype(str),
                        "query": new.query_id.map(qtext), "title": new.title, "genres": new.genres,
                        "plot_snippet": new.movie_id.astype(str).map(plot).fillna(""),
                        "relevance": "", "reviewer": "", "notes": ""})
    out = pd.concat([old, new], ignore_index=True)[LABEL_COLUMNS].sort_values(["query_id", "movie_id"], kind="stable")
    out.to_csv(JUDGMENTS, index=False, encoding="utf-8-sig")
    return len(new)


# --------------------------------------------------------------------------- metrics

# metrics_at_k imported from src.metrics


def read_judgments(path: Path) -> pd.DataFrame:
    """Read a judgment sheet saved with ',' or ';' (Excel), in UTF-8 or cp1250 (Czech Excel)."""
    if not path.is_file():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    raw = path.read_bytes()
    try:
        text, enc = raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, enc = raw.decode("cp1250"), "cp1250"
    header = text.splitlines()[0]
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=enc)


def load_grades(*paths: Path) -> dict:
    """(query_id, movie_id) -> grade 0/1/2 from several sheets; earlier paths win on overlap."""
    grade: dict = {}
    for path in reversed(paths):
        df = read_judgments(path)
        r = df.get("relevance", pd.Series(dtype=str)).astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        ok = r.isin(["0", "1", "2"])
        grade.update({(q, str(m)): int(g) for q, m, g in zip(df.loc[ok, "query_id"], df.loc[ok, "movie_id"], r[ok])})
    return grade


def metrics(top_k: int = 5) -> None:
    grade = load_grades(TFIDF_JUDGMENTS, JUDGMENTS)
    print(f"Loaded {len(grade)} graded query/movie pairs from {TFIDF_JUDGMENTS.name} + {JUDGMENTS.name}")

    def _find_variant_path(n: str) -> Path:
        p = SEMANTIC_DIR / f"{n}_mood_rankings.csv"
        if not p.is_file() and (RESULTS / f"{n}_mood_rankings.csv").is_file():
            return RESULTS / f"{n}_mood_rankings.csv"
        return p

    files = {"tfidf": _find_tfidf_rankings(), **{n: _find_variant_path(n) for n in VARIANTS}}
    per_query = []
    for method, path in files.items():
        if not path.is_file():
            continue
        rk = pd.read_csv(path, dtype={"movie_id": str})
        for qid, grp in rk.sort_values("rank").groupby("query_id"):
            g = [grade.get((qid, m)) for m in grp.movie_id.head(top_k)]
            row = {"method": method, "query_id": qid, "returned": len(g),
                   "judged": sum(x is not None for x in g)}
            if len(g) == top_k and all(x is not None for x in g):
                row |= metrics_at_k(g, top_k)
            per_query.append(row)
    pq = pd.DataFrame(per_query)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    pq.to_csv(COMPARISON_DIR / "comparison_per_query.csv", index=False, encoding="utf-8-sig")
    done = pq.dropna(subset=["precision@5"]) if "precision@5" in pq and not pq.empty else pq.iloc[0:0]
    if done.empty:
        print(f"No fully judged queries yet. Fill `relevance` in {JUDGMENTS.name} first.")
        if "method" in pq.columns:
            print(pq.groupby("method")[["judged"]].sum())
        return
    # Compare only on queries judged for EVERY method, otherwise means are not comparable.
    common = set.intersection(*[set(g.query_id) for _, g in done.groupby("method")])
    summary = (done[done.query_id.isin(common)].groupby("method")[["precision@5", "mrr@5", "ndcg@5"]]
               .mean().round(3).assign(n_queries=len(common)))
    summary.to_csv(COMPARISON_DIR / "comparison_metrics.csv", encoding="utf-8-sig")
    print(f"Metrics on {len(common)} queries judged for all methods:\n{summary}")


def _set_paths(results_dir: Path | None, judgments: Path | None) -> None:
    """Redirect outputs (e.g. for the full dataset) so sample results and labels are never overwritten."""
    global RESULTS, JUDGMENTS, TFIDF_RANKINGS, SEMANTIC_DIR, COMPARISON_DIR, TFIDF_DIR
    if results_dir is not None:
        RESULTS = results_dir if results_dir.is_absolute() else ROOT / results_dir
        SEMANTIC_DIR = RESULTS / "semantic1"
        COMPARISON_DIR = RESULTS / "comparison"
        TFIDF_DIR = RESULTS / "tfidf"
        TFIDF_RANKINGS = _find_tfidf_rankings()
    if judgments is not None:
        JUDGMENTS = judgments if judgments.is_absolute() else ROOT / judgments


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--index", type=Path, required=True)
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    r.add_argument("--pool-variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="Variants whose results are added to the human-judgment sheet")
    r.add_argument("--reference", type=Path, default=REFERENCE, help="CSV with imdb_id;clean_plot for snippets")
    m = sub.add_parser("metrics")
    for sp in (r, m):
        sp.add_argument("--results-dir", type=Path, help="Default evaluation/results (sample). Use a separate folder for full data.")
        sp.add_argument("--judgments", type=Path, help="Semantic labeling sheet. Default evaluation/mood_judgment_semantic.csv; with `run` only used (and extended) when given explicitly.")
    a = p.parse_args()
    _set_paths(a.results_dir, a.judgments)
    if a.cmd == "run":
        run(a.index, a.top_k, a.variants, a.pool_variants, a.reference, extend_sheet=a.judgments is not None)
    else:
        metrics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
