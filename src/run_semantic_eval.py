"""Run the shared mood queries through semantic-search variants and compare them with TF-IDF.

Two sub-commands (run from the repository root):

    # 1) retrieve: write rankings for every variant, add new query/movie pairs to mood_judgments.csv
    python -m src.run_semantic_eval run --index models/semantic_sample

    # 2) after humans filled `relevance` (0/1/2) in evaluation/mood_judgments.csv
    python -m src.run_semantic_eval metrics

    # full data: keep everything separate from the sample results that are being judged
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

ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "evaluation" / "mood_queries.json"
RESULTS = ROOT / "evaluation" / "results"
JUDGMENTS = ROOT / "evaluation" / "mood_judgments.csv"
REFERENCE = ROOT / "data" / "processed" / "sample_clean_movies.csv"
TFIDF_RANKINGS = RESULTS / "tfidf_mood_rankings.csv"

VARIANTS = {
    "sem_meta":      dict(source="meta"),
    "sem_subs_max":  dict(source="subtitles", aggregation="max"),
    "sem_subs_top3": dict(source="subtitles", aggregation="topk_mean", agg_k=3),
    "sem_combined":  dict(source="combined", aggregation="topk_mean", agg_k=3, meta_weight=0.5),
}
LABEL_COLUMNS = ["query_id", "movie_id", "query", "title", "genres", "plot_snippet", "relevance", "reviewer", "notes"]


# --------------------------------------------------------------------------- retrieval

def run(index: Path, top_k: int, variants: list[str], pool_variants: list[str], reference: Path) -> None:
    model = SemanticMovieRecommender.load(index)
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)

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
        rankings[rankings.method == name].to_csv(RESULTS / f"{name}_mood_rankings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(status_rows).to_csv(RESULTS / "semantic_query_status.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "semantic_mood_results.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved rankings for {len(variants)} variants to {RESULTS.relative_to(ROOT)}")

    overlap = overlap_table(rankings, top_k)
    if overlap is not None:
        overlap.to_csv(RESULTS / "method_overlap.csv", index=False, encoding="utf-8-sig")
        print("\nMean overlap of top-k lists with TF-IDF (share of shared movies):")
        print(overlap.drop(columns="query_id").mean(numeric_only=True).round(3).to_string())

    added = extend_judgments(rankings[rankings.method.isin(pool_variants)], queries, reference)
    print(f"\nAdded {added} new query/movie pairs to {JUDGMENTS.relative_to(ROOT)} (existing labels kept).")


def overlap_table(rankings: pd.DataFrame, top_k: int) -> pd.DataFrame | None:
    """|top-k(semantic) ∩ top-k(tfidf)| / k per query — computable before any human labels."""
    if not TFIDF_RANKINGS.is_file():
        print("No TF-IDF rankings found; skipping overlap.")
        return None
    tf = pd.read_csv(TFIDF_RANKINGS, dtype={"movie_id": str})
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

def metrics_at_k(grades, k=5) -> dict[str, float]:
    """Same definitions as notebooks/02_mood_query_evaluation.ipynb (judged-pool nDCG)."""
    vals = np.asarray(grades, dtype=int)
    binary = vals >= 1
    pos = np.flatnonzero(binary)
    disc = np.log2(np.arange(2, k + 2))
    dcg = float(np.sum((2.0 ** vals - 1) / disc))
    idcg = float(np.sum((2.0 ** np.sort(vals)[::-1] - 1) / disc))
    return {"precision@5": float(binary.mean()), "mrr@5": float(1 / (pos[0] + 1)) if len(pos) else 0.0,
            "ndcg@5": dcg / idcg if idcg else 0.0}


def metrics(top_k: int = 5) -> None:
    labels = pd.read_csv(JUDGMENTS, encoding="utf-8-sig", dtype={"movie_id": str})
    labels["relevance"] = pd.to_numeric(labels["relevance"], errors="coerce")
    grade = {(q, m): r for q, m, r in zip(labels.query_id, labels.movie_id, labels.relevance) if not np.isnan(r)}

    files = {"tfidf": TFIDF_RANKINGS, **{n: RESULTS / f"{n}_mood_rankings.csv" for n in VARIANTS}}
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
    pq.to_csv(RESULTS / "comparison_per_query.csv", index=False, encoding="utf-8-sig")
    done = pq.dropna(subset=["precision@5"]) if "precision@5" in pq else pq.iloc[0:0]
    if done.empty:
        print(f"No fully judged queries yet. Fill `relevance` in {JUDGMENTS.relative_to(ROOT) if JUDGMENTS.is_relative_to(ROOT) else JUDGMENTS} first.")
        print(pq.groupby("method")[["judged"]].sum())
        return
    # Compare only on queries judged for EVERY method, otherwise means are not comparable.
    common = set.intersection(*[set(g.query_id) for _, g in done.groupby("method")])
    summary = (done[done.query_id.isin(common)].groupby("method")[["precision@5", "mrr@5", "ndcg@5"]]
               .mean().round(3).assign(n_queries=len(common)))
    summary.to_csv(RESULTS / "comparison_metrics.csv", encoding="utf-8-sig")
    print(f"Metrics on {len(common)} queries judged for all methods:\n{summary}")


def _set_paths(results_dir: Path | None, judgments: Path | None) -> None:
    """Redirect outputs (e.g. for the full dataset) so sample results and labels are never overwritten."""
    global RESULTS, JUDGMENTS, TFIDF_RANKINGS
    if results_dir is not None:
        RESULTS = results_dir if results_dir.is_absolute() else ROOT / results_dir
        TFIDF_RANKINGS = RESULTS / "tfidf_mood_rankings.csv"
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
        sp.add_argument("--judgments", type=Path, help="Default evaluation/mood_judgments.csv")
    a = p.parse_args()
    _set_paths(a.results_dir, a.judgments)
    if a.cmd == "run":
        run(a.index, a.top_k, a.variants, a.pool_variants, a.reference)
    else:
        metrics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
