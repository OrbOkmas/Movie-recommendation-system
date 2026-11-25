"""Build docs/semantic_v2_results.md (+ CSVs) from rankings and human/LLM relevance judgments.

Run from the repository root, after `src.semantic_v2 run ... --name v2_A_sample` and after the judgments sheet
`evaluation/mood_judgments_v2.csv` is filled:

    python scripts/make_results_report.py

Everything in the report is computed here; nothing is typed in by hand. Methods that returned no results for a
query (abstention, empty query) are excluded from the *common* query set, and the exclusions are listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_v2.evaluate import agreement, bootstrap_ci, hub_stats, load_judgments, query_metrics  # noqa: E402

DEFAULT_RANKINGS = {
    "tfidf": "evaluation/results/tfidf_mood_rankings.csv",
    "sem_meta": "evaluation/results/sem_meta_mood_rankings.csv",
    "sem_subs_max": "evaluation/results/sem_subs_max_mood_rankings.csv",
    "sem_subs_top3": "evaluation/results/sem_subs_top3_mood_rankings.csv",
    "sem_combined": "evaluation/results/sem_combined_mood_rankings.csv",
    "v2": "evaluation/results/v2/v2_A_sample_rankings.csv",
}
METRICS = ["precision@k", "mrr@k", "ndcg@k_team", "ndcg@k_pool"]
SOURCES = ["plot_score", "keywords_score", "genres_score", "subtitle_score"]
ALIASES = {"movie_id": ["movie_id", "imdb_id", "id"], "query_id": ["query_id", "qid"], "rank": ["rank", "position"]}


def read_rankings(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    low = {c.lower().strip(): c for c in df.columns}
    for target, names in ALIASES.items():
        for n in names:
            if n in low:
                if low[n] != target:
                    df = df.rename(columns={low[n]: target})
                break
    for need in ("movie_id", "query_id"):
        if need not in df.columns:
            raise SystemExit(f"{path}: no column for {need} (found {list(df.columns)})")
    if "rank" not in df.columns:
        df["rank"] = df.groupby("query_id").cumcount() + 1
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["movie_id"] = df["movie_id"].astype(str)
    return df.dropna(subset=["rank"]).sort_values(["query_id", "rank"]).reset_index(drop=True)


def per_query(rankings: dict[str, pd.DataFrame], grades, k: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows, dropped = [], {}
    for method, rk in rankings.items():
        for qid, g in rk.groupby("query_id"):
            m = query_metrics(g.sort_values("rank").movie_id.tolist(), str(qid), grades, k)
            if m is None:
                dropped.setdefault(method, []).append(f"{qid} (not fully judged)")
            else:
                rows.append({"method": method, "query_id": str(qid), **m})
    return pd.DataFrame(rows), dropped


def summarize(pq: pd.DataFrame, queries: list[str]) -> pd.DataFrame:
    out = []
    sub = pq[pq.query_id.isin(queries)]
    for method, g in sub.groupby("method", sort=False):
        row = {"method": method, "queries": len(g)}
        for col in METRICS:
            lo, hi = bootstrap_ci(g[col].to_numpy())
            row[col] = f"{g[col].mean():.3f} [{lo:.2f}, {hi:.2f}]"
        out.append(row)
    return pd.DataFrame(out)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", default="evaluation/mood_judgments_v2.csv")
    ap.add_argument("--queries", default="evaluation/mood_queries.json")
    ap.add_argument("--rankings", nargs="*", help="name=path (default: tfidf, 4 semantic v1 variants, v2)")
    ap.add_argument("--main", default="v2", help="the method the analysis is about")
    ap.add_argument("--vague", nargs="*", default=["M25", "M27"], help="queries without criteria, reported separately")
    ap.add_argument("--human", help="judgments of a human annotator on the validation subset (for kappa)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-md", default="docs/semantic_v2_results.md")
    ap.add_argument("--out-dir", default="evaluation/results/v2")
    a = ap.parse_args()

    specs = dict(s.split("=", 1) for s in a.rankings) if a.rankings else DEFAULT_RANKINGS
    rankings = {}
    for name, p in specs.items():
        if not Path(p).is_file():
            print(f"skipping {name}: {p} not found")
            continue
        rankings[name] = read_rankings(p)
    if a.main not in rankings:
        raise SystemExit(f"main method {a.main!r} has no rankings file")
    grades = load_judgments(a.judgments)
    jdf = pd.read_csv(a.judgments, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    snippet = {(r.query_id, r.movie_id): r.plot_snippet for r in jdf.itertuples()}
    reviewers = jdf.reviewer.replace("", "(empty)").value_counts().to_dict()
    qmeta = {str(q.get("query_id", q.get("id"))): q for q in json.loads(Path(a.queries).read_text(encoding="utf-8"))}
    cat = {k: v.get("category", "") for k, v in qmeta.items()}

    pq, dropped = per_query(rankings, grades, a.k)
    judged_queries = {q for q, _ in grades}
    for method, rk in rankings.items():
        for q in sorted(judged_queries - set(rk.query_id.unique())):
            dropped.setdefault(method, []).append(f"{q} (returned no results)")
    common = sorted(set.intersection(*[set(g.query_id) for _, g in pq.groupby("method")]))
    common_nv = [q for q in common if q not in a.vague]
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    pq[pq.query_id.isin(common)].to_csv(Path(a.out_dir) / "sample_per_query_metrics.csv", index=False)

    L = [f"# Semantic search v2 - results on the 100-movie sample\n",
         "_Generated by `scripts/make_results_report.py`; do not edit by hand._\n",
         "## Setup\n",
         f"- Corpus: the 100-movie sample used for all methods; {len(qmeta)} mood queries (M01-M28); top-{a.k} per query.",
         f"- Judged pairs: {len(grades)} (relevance 0/1/2). Labelers in the sheet: {reviewers}.",
         "- Metrics: P@k (relevance >= 1), MRR@k, nDCG@k as defined in the team notebook (`ndcg@k_team`: ideal ranking taken "
         "from the retrieved list itself) and `ndcg@k_pool` (ideal ranking taken from every judged movie of the query). "
         "Brackets are 95% bootstrap intervals over queries.",
         f"- Queries judged for **all** methods: {len(common)}. Excluded for a method = it returned nothing or a top-{a.k} "
         "movie is unjudged:"]
    for m, qs in dropped.items():
        L.append(f"  - {m}: {', '.join(sorted(qs))}")
    if a.human:
        ag = agreement(a.judgments, a.human)
        L.append(f"- Agreement with the human validation subset: {json.dumps(ag)}")
    L += ["\n## Overall\n", md_table(summarize(pq, common)),
          f"\nWithout the vague queries ({', '.join(a.vague)}), {len(common_nv)} queries:\n",
          md_table(summarize(pq, common_nv))]
    summ = summarize(pq, common_nv)
    summ.to_csv(Path(a.out_dir) / "sample_summary.csv", index=False)

    # per category
    bycat = pq[pq.query_id.isin(common_nv)].assign(category=lambda d: d.query_id.map(cat))
    t = bycat.groupby(["category", "method"])["ndcg@k_pool"].mean().unstack("method").round(2)
    t.insert(0, "n_queries", bycat.groupby("category").query_id.nunique())
    L += ["\n## By query category (mean nDCG@k pool, vague queries excluded)\n", md_table(t.reset_index())]

    # paired differences
    L.append("\n## Paired difference: main method minus each other method (vague queries excluded)\n")
    rows = []
    wide = {c: pq[pq.query_id.isin(common_nv)].pivot(index="query_id", columns="method", values=c) for c in ("ndcg@k_pool", "precision@k")}
    for other in [m for m in rankings if m != a.main]:
        d = (wide["ndcg@k_pool"][a.main] - wide["ndcg@k_pool"][other]).to_numpy()
        dp = (wide["precision@k"][a.main] - wide["precision@k"][other]).to_numpy()
        lo, hi = bootstrap_ci(d)
        rows.append({"vs": other, "mean diff nDCG pool": f"{d.mean():+.3f} [{lo:+.2f}, {hi:+.2f}]",
                     "mean diff P@k": f"{dp.mean():+.3f}", "wins/ties/losses": f"{(d > 1e-9).sum()}/{(abs(d) <= 1e-9).sum()}/{(d < -1e-9).sum()}"})
    L.append(md_table(pd.DataFrame(rows)))

    # label-free: overlap and hubs
    L.append("\n## Label-free diagnostics\n")
    sets = {m: rk.groupby("query_id").movie_id.apply(set) for m, rk in rankings.items()}
    rows = []
    for m, rk in rankings.items():
        h = hub_stats(rk)
        ov = [len(sets[a.main][q] & sets[m][q]) / a.k for q in sets[a.main].index.intersection(sets[m].index)] if m != a.main else []
        rows.append({"method": m, "queries with results": h["queries"], "distinct movies in top-k": h["distinct_movies"],
                     "most repeated movie (times)": h["max_appearances"], f"mean overlap@{a.k} with {a.main}": f"{np.mean(ov):.2f}" if ov else "-"})
    L.append(md_table(pd.DataFrame(rows)))

    # do the sources separate relevant from irrelevant results? (the project's key question about subtitles)
    v2 = rankings[a.main]
    v2["relevance"] = [grades.get((q, m)) for q, m in zip(v2.query_id, v2.movie_id)]
    present = [c for c in SOURCES if c in v2.columns]
    if present:
        d = v2.dropna(subset=["relevance"]).copy()
        for c in present:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        tab = d.groupby("relevance")[present].mean().round(3)
        tab.insert(0, "n", d.groupby("relevance").size())
        tab.index = tab.index.astype(int)
        tab["n"] = tab["n"].astype(int)
        L += [f"\n## Raw similarity of each source for the retrieved results of `{a.main}`, by relevance label\n",
              "Cosines of different sources are not comparable with each other, but within a column a source helps "
              "if relevant results (1, 2) score higher than irrelevant ones (0).\n", md_table(tab.reset_index())]

    # error analysis: worst queries of the main method
    worst = pq[(pq.method == a.main) & pq.query_id.isin(common_nv)].sort_values(["ndcg@k_pool", "precision@k"]).head(5)
    L.append("\n## Error analysis: the five weakest queries of the main method\n")
    for _, r in worst.iterrows():
        g = v2[v2.query_id == r.query_id].sort_values("rank")
        L.append(f"\n### {r.query_id} ({cat.get(r.query_id, '')}): {qmeta.get(r.query_id, {}).get('query', '')}\n")
        L.append(f"P@{a.k} = {r['precision@k']:.2f}, nDCG pool = {r['ndcg@k_pool']:.2f}\n")
        rows = []
        for x in g.to_dict("records"):
            rows.append({"rank": int(x["rank"]), "title": x.get("title", x["movie_id"]), "genres": x.get("genres", ""),
                         "label": grades.get((x["query_id"], x["movie_id"]), "?"),
                         **{"cos_" + c.replace("_score", ""): x.get(c, "") for c in present},
                         "plot": snippet.get((x["query_id"], x["movie_id"]), "")[:110]})
        L.append(md_table(pd.DataFrame(rows)))
    L += ["\n## Threats to validity\n",
          "- 100 movies and 27 judged queries: differences between methods are within noise unless the intervals above exclude zero.",
          "- Most pooled candidates are poor matches in a 100-movie corpus, so absolute scores are low for every method.",
          "- Labels come from a first pass by an AI assistant (reviewer `claude`) and are validated on a random subset by humans "
          "(see the agreement line above); both can be wrong, and an AI judge may prefer semantically plausible matches.",
          "- The design of v2 was informed by documented failures on these same queries (negation, typos, vague queries); "
          "the 50 dev queries (`evaluation/dev_queries.json`) were written separately for tuning, so this sample result is "
          "not an untouched test.",
          "- Vague queries have no criteria; they are scored 1 for every film and reported separately."]
    Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {a.out_md} and CSVs in {a.out_dir}")


if __name__ == "__main__":
    main()
