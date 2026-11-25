import numpy as np
import pandas as pd
import pytest

from semantic_v2.evaluate import (agreement, bootstrap_ci, cohen_kappa, hub_stats, labeled_summary, query_metrics, stability, status_accuracy,
                                  suggest_threshold, weak_genre_hit_rate)


def test_ndcg_pool_penalises_weak_lists_but_team_version_does_not():
    grades = {("q", "a"): 1, ("q", "b"): 1, ("q", "c"): 1, ("q", "d"): 1, ("q", "e"): 1, ("q", "z1"): 2, ("q", "z2"): 2}
    m = query_metrics(["a", "b", "c", "d", "e"], "q", grades)
    assert m["ndcg@k_team"] == pytest.approx(1.0)             # the team definition cannot see the missed 2s
    assert m["ndcg@k_pool"] < 0.7
    best = query_metrics(["z1", "z2", "a", "b", "c"], "q", grades)
    assert best["ndcg@k_pool"] == pytest.approx(1.0)


def test_unjudged_top_k_is_skipped():
    assert query_metrics(["a", "b"], "q", {("q", "a"): 1}) is None


def test_metrics_basic():
    g = {("q", "a"): 0, ("q", "b"): 2, ("q", "c"): 1}
    m = query_metrics(["a", "b", "c"], "q", g, k=3)
    assert m["precision@k"] == pytest.approx(2 / 3) and m["mrr@k"] == pytest.approx(0.5)


def test_summary_uses_common_queries_and_ci():
    g = {("q1", "a"): 2, ("q1", "b"): 0, ("q2", "a"): 1, ("q2", "b"): 0}
    r1 = pd.DataFrame({"query_id": ["q1", "q1", "q2", "q2"], "rank": [1, 2, 1, 2], "movie_id": list("abab")})
    r2 = pd.DataFrame({"query_id": ["q1", "q1"], "rank": [1, 2], "movie_id": list("ba")})
    per_q, s = labeled_summary({"m1": r1, "m2": r2}, g, k=2)
    assert set(per_q.query_id) == {"q1"} and set(s.n_queries) == {1}
    lo, hi = bootstrap_ci(np.array([0.2, 0.4, 0.6]))
    assert lo <= 0.4 <= hi


def test_hub_stats():
    rk = pd.DataFrame({"query_id": list("aabbcc"), "movie_id": ["x", "y", "x", "z", "x", "w"]})
    s = hub_stats(rk)
    assert s["max_appearances"] == 3 and s["most_frequent"] == "x" and s["distinct_movies"] == 4


def test_stability_and_weak_genre_and_status():
    q = [{"query_id": "A"}, {"query_id": "B", "typo_of": "A", "weak_genre_expect": ["Horror"], "expected_status": "ok"}]
    rk = pd.DataFrame({"query_id": ["A", "A", "B", "B"], "movie_id": ["1", "2", "2", "3"],
                       "genres": ["Drama", "Horror|Drama", "Horror", "Comedy"]})
    assert stability(rk, q)["overlap@k"].iloc[0] == pytest.approx(0.5)
    assert weak_genre_hit_rate(rk, q)["hit_rate@k"].iloc[0] == 0.5
    st = pd.DataFrame({"query_id": ["B"], "status": ["ok"]})
    assert status_accuracy(st, q).ok.all()


def test_suggest_threshold():
    r = suggest_threshold([0.1, 0.2], [0.4, 0.5, 0.6])
    assert r["separable"] and 0.2 < r["threshold"] <= 0.4


def test_kappa_perfect_chance_and_weighted():
    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    near, far = [0, 1, 2, 0, 1, 2], [1, 1, 2, 0, 2, 2], 
    assert cohen_kappa(near, far, "linear") > cohen_kappa(near, far)


def test_agreement_files(tmp_path):
    a = pd.DataFrame({"query_id": ["q"] * 4, "movie_id": list("abcd"), "relevance": [0, 1, 2, 0]})
    b = pd.DataFrame({"query_id": ["q"] * 4, "movie_id": list("abcd"), "relevance": [0, 1, 1, 1]})
    a.to_csv(tmp_path / "a.csv", index=False); b.to_csv(tmp_path / "b.csv", index=False)
    r = agreement(tmp_path / "a.csv", tmp_path / "b.csv")
    assert r["n"] == 4 and r["exact_agreement"] == 0.5 and r["binary_relevant_agreement"] == 0.75


def test_cli_reads_rankings_with_other_column_names(tmp_path):
    from semantic_v2.cli import _read_rankings
    pd.DataFrame({"QID": ["q", "q"], "imdb_id": ["a", "b"]}).rename(columns={"QID": "qid"}).to_csv(tmp_path / "r.csv", index=False)
    df = _read_rankings(tmp_path / "r.csv")
    assert list(df["rank"]) == [1, 2] and set(df.columns) >= {"query_id", "movie_id", "rank"}
