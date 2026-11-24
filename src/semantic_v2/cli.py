"""Command line: python -m src.semantic_v2 {train,search,run,metrics,diag,calibrate,profile} ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_REGISTRY, SOURCES, BuildConfig, QueryConfig, SearchConfig, SemanticV2Error
from .evaluate import (agreement, hub_stats, labeled_summary, load_judgments, load_queries, status_accuracy, stability,
                       suggest_threshold, weak_genre_hit_rate)
from .fields import LoadStats, iter_frames, iter_movies, is_missing

LOGGER = logging.getLogger("semantic_v2")
LABEL_COLUMNS = ["query_id", "movie_id", "query", "title", "genres", "plot_snippet", "relevance", "reviewer", "notes"]


# ------------------------------------------------------------------------------------------- arg helpers
def add_search_args(p: argparse.ArgumentParser) -> None:
    d = SearchConfig()
    p.add_argument("--config", help="SearchConfig JSON (e.g. configs/v2_default.json); overrides the flags below")
    for s in SOURCES:
        p.add_argument(f"--w-{s}", type=float, default=d.weights[s], help=f"weight of the {s} source")
    p.add_argument("--hub-fields", nargs="*", default=list(d.hub_fields), choices=SOURCES,
                   help="sources with hub correction (empty list = off)")
    p.add_argument("--hub-gamma", type=float, default=d.hub_gamma)
    p.add_argument("--alpha", type=float, default=d.alpha, help="whole-query weight vs. soft-min over clauses")
    p.add_argument("--softmin-t", type=float, default=d.softmin_T)
    p.add_argument("--lam", type=float, default=d.lam, help="negation penalty strength (0 = ignore negation)")
    p.add_argument("--tau", type=float, default=d.tau)
    p.add_argument("--no-clauses", action="store_true")
    p.add_argument("--meta-negation", action="store_true", help="D8: genre/keyword based negation")
    p.add_argument("--min-confidence", type=float, default=d.min_confidence)
    p.add_argument("--no-spell", action="store_true")


def search_cfg(a) -> SearchConfig:
    if getattr(a, "config", None):
        return SearchConfig.from_json(a.config)
    return SearchConfig(weights={s: getattr(a, f"w_{s}") for s in SOURCES}, hub_fields=tuple(a.hub_fields),
                        hub_gamma=a.hub_gamma, alpha=a.alpha, softmin_T=a.softmin_t, lam=a.lam, tau=a.tau,
                        use_clauses=not a.no_clauses, meta_negation=a.meta_negation, min_confidence=a.min_confidence)


def query_cfg(a) -> QueryConfig:
    clauses = search_cfg(a).use_clauses if getattr(a, "config", None) else not a.no_clauses
    return QueryConfig(spell=not a.no_spell, clauses=clauses)


def _searcher(a):
    from .search import SemanticSearcher
    return SemanticSearcher.from_dir(a.index, search_cfg(a), query_cfg(a), device=getattr(a, "device", None))


# ------------------------------------------------------------------------------------------- commands
def cmd_train(a) -> None:
    from .index import build_index
    cfg = BuildConfig(field_model=a.field_model, chunk_model=a.chunk_model, chunk_words=a.chunk_words,
                      max_chunks_per_movie=a.max_chunks_per_movie, shard_movies=a.shard_movies, batch_size=a.batch_size)
    exclude = [q["query"] for q in load_queries(*a.exclude_queries)] if a.exclude_queries else []
    m = build_index(a.data, a.output, cfg, max_rows=a.max_rows, overwrite=a.overwrite,
                    store_chunk_text=not a.no_chunk_text, device=a.device, exclude_texts=exclude)
    print(json.dumps(m["stats"], indent=2))


def _print_response(r) -> None:
    q = r.query
    print(f"status: {r.status}" + (f"  confidence: {r.confidence}" if r.confidence is not None else ""))
    if q.get("corrections"):
        print(f"  corrected: {q['corrections']}  -> '{q['text']}'")
    if q.get("neg_terms"):
        print(f"  understood as: want '{q.get('whole_text')}'  but NOT {q['neg_terms']}")
    if q.get("clauses"):
        print(f"  clauses: {q['clauses']}")
    for w in r.warnings:
        print(f"  warning [{w['code']}]: {w['message']}")
    if r.message:
        print("  " + r.message)
    for x in r.results:
        print(f"{x['rank']}. {x['title']} ({x['year']})  [{x['genres']}]  score={x['score']}")
        parts = [f"{k}={x[k]}" for k in ("plot_score", "keywords_score", "genres_score", "subtitle_score") if x.get(k) is not None]
        print("     " + "  ".join(parts) + (f"  neg_penalty={x['neg_penalty']}" if x["neg_penalty"] else ""))
        if x.get("best_chunk"):
            print(f"     dialogue: {x['best_chunk']}")


def cmd_search(a) -> None:
    s = _searcher(a)
    r = s.search(a.query, top_k=a.top_k, return_all=a.show_low_confidence)
    print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False) if a.json else "", end="")
    if not a.json:
        _print_response(r)


def cmd_run(a) -> None:
    s = _searcher(a)
    queries = load_queries(*a.queries)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows, status_rows, t0 = [], [], time.time()
    for q in queries:
        t1 = time.time()
        r = s.search(q["query"], top_k=a.top_k, return_all=True, snippets=False)
        status_rows.append({"method": a.name, "query_id": q["query_id"], "category": q.get("category", ""),
                            "query": q["query"][:200], "status": r.status, "confidence": r.confidence,
                            "n_warnings": len(r.warnings), "warning_codes": ";".join(w["code"] for w in r.warnings),
                            "latency_ms": round((time.time() - t1) * 1000)})
        for x in r.results:
            rows.append({"method": a.name, "query_id": q["query_id"], "category": q.get("category", ""),
                         "query": q["query"][:200], "status": r.status, **{k: x.get(k) for k in (
                             "rank", "movie_id", "title", "year", "genres", "score", "plot_score", "keywords_score",
                             "genres_score", "subtitle_score", "neg_penalty")}})
    rk = pd.DataFrame(rows)
    rk.to_csv(out / f"{a.name}_rankings.csv", index=False, encoding="utf-8")
    pd.DataFrame(status_rows).to_csv(out / f"{a.name}_query_status.csv", index=False, encoding="utf-8")
    print(f"{len(queries)} queries in {time.time() - t0:.1f}s -> {out}/{a.name}_rankings.csv")
    if a.judgments:
        n = extend_judgments(rk, queries, s.index.movies, Path(a.judgments))
        print(f"{n} new query/movie pairs added to {a.judgments} (fill `relevance` 0/1/2).")


def extend_judgments(rk: pd.DataFrame, queries: list[dict], movies: pd.DataFrame, path: Path) -> int:
    old = pd.read_csv(path, encoding="utf-8-sig", dtype={"movie_id": str}) if path.is_file() else pd.DataFrame(columns=LABEL_COLUMNS)
    seen = set(zip(old.query_id.astype(str), old.movie_id.astype(str)))
    qtext = {q["query_id"]: q["query"] for q in queries}
    snip = dict(zip(movies.imdb_id, movies.get("plot_snippet", pd.Series([""] * len(movies)))))
    new = rk.drop_duplicates(["query_id", "movie_id"])
    new = new[[(q, str(m)) not in seen for q, m in zip(new.query_id, new.movie_id)]]
    add = pd.DataFrame({"query_id": new.query_id, "movie_id": new.movie_id.astype(str), "query": new.query_id.map(qtext),
                        "title": new.title, "genres": new.genres, "plot_snippet": new.movie_id.map(snip).fillna(""),
                        "relevance": "", "reviewer": "", "notes": ""})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, add], ignore_index=True)[LABEL_COLUMNS].sort_values(["query_id", "movie_id"], kind="stable") \
        .to_csv(path, index=False, encoding="utf-8-sig")
    return len(add)


def cmd_metrics(a) -> None:
    grades = load_judgments(a.judgments)
    rankings = {}
    for spec in a.rankings:
        name, _, path = spec.partition("=") if "=" in spec else (Path(spec).stem.replace("_rankings", ""), "", spec)
        rankings[name] = pd.read_csv(path, dtype={"movie_id": str})
    per_q, summary = labeled_summary(rankings, grades, a.k)
    if summary.empty:
        print("No query is fully judged for every method yet. Fill `relevance` in the judgments file first.")
        return
    print(f"Queries judged for ALL methods: {summary.n_queries.iloc[0]}\n{summary.to_string(index=False)}")
    if a.out:
        per_q.to_csv(a.out, index=False)


def cmd_diag(a) -> None:
    queries = load_queries(*a.queries)
    rk = pd.read_csv(a.rankings, dtype={"movie_id": str})
    rk_by_q = rk.copy()
    print("HUB CONCENTRATION", json.dumps(hub_stats(rk_by_q)))
    stab = stability(rk_by_q.assign(), queries)
    if not stab.empty:
        print("\nSTABILITY (overlap@k of top results)\n" + stab.to_string(index=False))
    g = weak_genre_hit_rate(rk_by_q, queries)
    if not g.empty:
        print(f"\nWEAK GENRE HIT RATE (sanity only): mean={g['hit_rate@k'].mean():.2f} over {len(g)} queries")
    if a.status:
        st = pd.read_csv(a.status)
        acc = status_accuracy(st, queries)
        if not acc.empty:
            print(f"\nSTATUS ACCURACY: {acc.ok.mean():.2f} ({int(acc.ok.sum())}/{len(acc)})")
            print(acc[~acc.ok].to_string(index=False))
        lat = pd.to_numeric(st.latency_ms, errors="coerce")
        print(f"\nLATENCY ms: median={lat.median():.0f} p95={lat.quantile(.95):.0f}")


def cmd_calibrate(a) -> None:
    s = _searcher(a)
    low, normal = [], []
    for q in load_queries(*a.queries):
        exp = q.get("expected_status")
        r = s.search(q["query"], return_all=True, snippets=False)
        if r.confidence is None:
            continue
        if exp == "low_confidence":
            low.append(r.confidence)
        elif q.get("tier") == "core" or exp in {"ok", "ok_with_warning"}:
            normal.append(r.confidence)
        print(f"{q['query_id']:>4} conf={r.confidence:.3f} expected={exp or 'ok'}  {q['query'][:60]!r}")
    print("\nSuggestion:", json.dumps(suggest_threshold(low, normal)))
    print("Use it with --min-confidence <threshold>. Re-check after every change of weights or models.")


def cmd_agreement(a) -> None:
    r = agreement(a.a, a.b)
    print(json.dumps(r, indent=2))
    if r.get("n", 0) and r["kappa_binary"] < 0.6:
        print("Note: binary kappa below 0.6 - discuss the disagreements and refine the labeling guide.")


def cmd_profile(a) -> None:
    cfg = BuildConfig()
    stats = LoadStats()
    plot_w, kw_n, years, dlg_w, genres = [], [], [], [], {}
    for rec in iter_movies(a.data, cfg, stats, max_rows=a.max_rows):
        plot_w.append(len(rec.plot.split()) if rec.plot else 0)
        kw_n.append(len([k for k in rec.keywords_raw.split("|") if k]))
        dlg_w.append(sum(len(c.split()) for c in rec.chunks))
        if rec.year.isdigit():
            years.append(int(rec.year))
        for g in [x for x in rec.genres_raw.split("|") if x]:
            genres[g] = genres.get(g, 0) + 1
    n = max(stats.kept, 1)
    q = lambda arr, p: float(np.percentile(arr, p)) if len(arr) else float("nan")  # noqa: E731
    report = {
        "rows_read": stats.rows_read, "duplicate_ids": stats.duplicate_ids, "excluded_no_text": stats.excluded_no_text,
        "rankable_movies": stats.kept,
        "missing_pct": {"plot": round(100 * stats.no_plot / n, 1), "keywords": round(100 * stats.no_keywords / n, 1),
                        "genres": round(100 * stats.no_genres / n, 1), "dialogue": round(100 * stats.no_dialogue / n, 1)},
        "plot_words": {p: round(q([w for w in plot_w if w], p), 1) for p in (5, 25, 50, 75, 95)},
        "keywords_per_movie": {p: round(q(kw_n, p), 1) for p in (50, 75, 95, 100)},
        "dialogue_words_in_chunks": {p: round(q([w for w in dlg_w if w], p), 0) for p in (5, 50, 95)},
        "est_chunks_total": int(sum(dlg_w) / cfg.chunk_words),
        "year": {"min": min(years) if years else None, "median": q(years, 50), "max": max(years) if years else None},
        "top_genres_pct": {g: round(100 * c / n, 1) for g, c in sorted(genres.items(), key=lambda kv: -kv[1])[:12]},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="semantic_v2", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="build (or resume) an index")
    t.add_argument("--data", required=True); t.add_argument("--output", required=True)
    t.add_argument("--field-model", default="minilm-l6", choices=sorted(MODEL_REGISTRY))
    t.add_argument("--chunk-model", default="minilm-l6", choices=sorted(MODEL_REGISTRY))
    t.add_argument("--chunk-words", type=int, default=150); t.add_argument("--max-chunks-per-movie", type=int, default=0)
    t.add_argument("--shard-movies", type=int, default=2000); t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--max-rows", type=int); t.add_argument("--overwrite", action="store_true")
    t.add_argument("--no-chunk-text", action="store_true"); t.add_argument("--device")
    t.add_argument("--exclude-queries", nargs="*", help="query JSON files whose texts must not enter the hub bank")
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("search", help="search one query")
    s.add_argument("--index", required=True); s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5); s.add_argument("--json", action="store_true")
    s.add_argument("--show-low-confidence", action="store_true"); s.add_argument("--device")
    add_search_args(s); s.set_defaults(fn=cmd_search)

    r = sub.add_parser("run", help="run query files -> rankings/status CSVs (+ judgment sheet)")
    r.add_argument("--index", required=True); r.add_argument("--queries", nargs="+", required=True)
    r.add_argument("--name", required=True); r.add_argument("--output-dir", default="evaluation/results/v2")
    r.add_argument("--judgments", help="judgment sheet to extend with unjudged pairs (use a v2-specific file)")
    r.add_argument("--top-k", type=int, default=5); r.add_argument("--device")
    add_search_args(r); r.set_defaults(fn=cmd_run)

    m = sub.add_parser("metrics", help="labeled metrics with bootstrap CI")
    m.add_argument("--rankings", nargs="+", required=True, help="name=path or path")
    m.add_argument("--judgments", required=True); m.add_argument("--k", type=int, default=5); m.add_argument("--out")
    m.set_defaults(fn=cmd_metrics)

    d = sub.add_parser("diag", help="label-free diagnostics for one run")
    d.add_argument("--rankings", required=True); d.add_argument("--queries", nargs="+", required=True)
    d.add_argument("--status"); d.set_defaults(fn=cmd_diag)

    c = sub.add_parser("calibrate", help="suggest the abstention threshold from dev queries")
    c.add_argument("--index", required=True); c.add_argument("--queries", nargs="+", required=True); c.add_argument("--device")
    add_search_args(c); c.set_defaults(fn=cmd_calibrate)

    ag = sub.add_parser("agreement", help="agreement (Cohen's kappa) between two judgment files")
    ag.add_argument("--a", required=True); ag.add_argument("--b", required=True); ag.set_defaults(fn=cmd_agreement)

    pr = sub.add_parser("profile", help="data profile (missing rates, lengths, genres) for the design decisions")
    pr.add_argument("--data", required=True); pr.add_argument("--max-rows", type=int); pr.add_argument("--out")
    pr.set_defaults(fn=cmd_profile)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except (SemanticV2Error, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
