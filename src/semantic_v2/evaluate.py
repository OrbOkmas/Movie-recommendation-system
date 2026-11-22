"""Evaluation: labeled metrics (team definitions + a corrected nDCG + bootstrap CI) and label-free diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_judgments(path: str | Path) -> dict[tuple[str, str], int]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"movie_id": str})
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce")
    df = df.dropna(subset=["relevance"])
    return {(str(q), str(m)): int(r) for q, m, r in zip(df.query_id, df.movie_id, df.relevance)}


def _dcg(vals, k) -> float:
    vals = np.asarray(vals, dtype=float)[:k]
    return float(np.sum((2.0 ** vals - 1) / np.log2(np.arange(2, len(vals) + 2))))


def query_metrics(ranked_ids: list[str], qid: str, grades: dict[tuple[str, str], int], k: int = 5) -> dict | None:
    """None if any of the top-k is unjudged (comparing partially judged queries would be unfair)."""
    g = [grades.get((qid, m)) for m in ranked_ids[:k]]
    if len(g) == 0 or any(x is None for x in g):
        return None
    vals = np.asarray(g, dtype=int)
    pos = np.flatnonzero(vals >= 1)
    idcg_own = _dcg(np.sort(vals)[::-1], k)
    pool = sorted((r for (q, _), r in grades.items() if q == qid), reverse=True)
    idcg_pool = _dcg(pool, k)
    return {"precision@k": float((vals >= 1).mean()), "mrr@k": float(1 / (pos[0] + 1)) if len(pos) else 0.0,
            "ndcg@k_team": _dcg(vals, k) / idcg_own if idcg_own else 0.0,       # ideal = own list (team definition)
            "ndcg@k_pool": _dcg(vals, k) / idcg_pool if idcg_pool else 0.0}      # ideal = best judged pool


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 13) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def labeled_summary(rankings: dict[str, pd.DataFrame], grades, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rankings: method -> DataFrame(query_id, rank, movie_id). Compares on queries judged for every method."""
    rows = []
    for method, rk in rankings.items():
        for qid, grp in rk.sort_values("rank").groupby("query_id"):
            m = query_metrics(grp.movie_id.astype(str).tolist(), str(qid), grades, k)
            if m:
                rows.append({"method": method, "query_id": qid, **m})
    per_q = pd.DataFrame(rows)
    if per_q.empty:
        return per_q, per_q
    common = set.intersection(*[set(g.query_id) for _, g in per_q.groupby("method")])
    per_q = per_q[per_q.query_id.isin(common)]
    out = []
    for method, g in per_q.groupby("method"):
        row = {"method": method, "n_queries": len(g)}
        for col in ("precision@k", "mrr@k", "ndcg@k_team", "ndcg@k_pool"):
            lo, hi = bootstrap_ci(g[col].to_numpy())
            row[col] = round(g[col].mean(), 3); row[col + "_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        out.append(row)
    return per_q, pd.DataFrame(out)


# ------------------------------------------------------------------------------ label-free diagnostics
def hub_stats(rk: pd.DataFrame) -> dict:
    counts = rk.groupby("movie_id").size().sort_values(ascending=False)
    n = len(counts)
    x = np.sort(counts.to_numpy(dtype=float))
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else 0.0
    top = counts.index[0] if n else None
    return {"queries": rk.query_id.nunique(), "slots": len(rk), "distinct_movies": n,
            "max_appearances": int(counts.iloc[0]) if n else 0, "most_frequent": top, "gini": round(gini, 3)}


def overlap_at_k(rk: pd.DataFrame, a: str, b: str) -> float | None:
    sets = rk.groupby("query_id")["movie_id"].apply(set)
    if a not in sets.index or b not in sets.index or not sets[a] or not sets[b]:
        return None
    return len(sets[a] & sets[b]) / max(len(sets[a]), len(sets[b]))


def stability(rk: pd.DataFrame, queries: list[dict]) -> pd.DataFrame:
    rows = []
    for q in queries:
        if q.get("typo_of"):
            rows.append({"kind": "typo", "a": q["typo_of"], "b": q["query_id"], "overlap@k": overlap_at_k(rk, q["typo_of"], q["query_id"])})
    groups: dict[str, list[str]] = {}
    for q in queries:
        if q.get("paraphrase_group"):
            groups.setdefault(q["paraphrase_group"], []).append(q["query_id"])
    for g, ids in groups.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                rows.append({"kind": "paraphrase:" + g, "a": ids[i], "b": ids[j], "overlap@k": overlap_at_k(rk, ids[i], ids[j])})
    return pd.DataFrame(rows)


def weak_genre_hit_rate(rk: pd.DataFrame, queries: list[dict]) -> pd.DataFrame:
    """Share of top-k whose genres contain at least one expected genre. A sanity check only (false negatives exist)."""
    rows = []
    for q in queries:
        exp = {g.lower() for g in q.get("weak_genre_expect", [])}
        if not exp:
            continue
        g = rk[rk.query_id == q["query_id"]]
        if g.empty:
            continue
        hit = g["genres"].fillna("").map(lambda s: bool(exp & {x.strip().lower() for x in str(s).split("|")}))
        rows.append({"query_id": q["query_id"], "hit_rate@k": round(float(hit.mean()), 2)})
    return pd.DataFrame(rows)


def status_accuracy(status: pd.DataFrame, queries: list[dict]) -> pd.DataFrame:
    exp = {q["query_id"]: q["expected_status"] for q in queries if q.get("expected_status")}
    rows = [{"query_id": r.query_id, "expected": exp[r.query_id], "actual": r.status, "ok": exp[r.query_id] == r.status}
            for r in status.itertuples() if r.query_id in exp]
    return pd.DataFrame(rows)


def suggest_threshold(low: list[float], normal: list[float]) -> dict:
    """Pick the abstention threshold on dev confidences: maximise (TPR - FPR) of "should abstain"."""
    if not low or not normal:
        return {"threshold": None, "note": "need both expected-abstain and normal queries"}
    cands = sorted(set(low + normal))
    best = (-2.0, None)
    for t in cands:
        tpr = np.mean([x < t for x in low]); fpr = np.mean([x < t for x in normal])
        if tpr - fpr > best[0]:
            best = (tpr - fpr, t)
    return {"threshold": round(float(best[1]), 4), "youden_j": round(float(best[0]), 3),
            "max_low": round(max(low), 4), "min_normal": round(min(normal), 4),
            "separable": max(low) < min(normal)}


def load_queries(*paths: str | Path) -> list[dict]:
    out = []
    for p in paths:
        out += json.loads(Path(p).read_text(encoding="utf-8"))
    return out
