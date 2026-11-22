"""Score computation: raw per-source cosines (stage A, expensive, cacheable) and calibrated fusion (stage B, cheap).

Stage B recipe (docs/semantic_v2_plan.md section 5):
    s'_f  = s_f - gamma * hubness_f(movie)                 (only for sources in ``hub_fields``)
    z_f   = (s'_f - mean_f(query)) / std_f(query)           (over all movies that have source f)
    z(m)  = sum_f w_f z_f / sum_f w_f                       (over the sources movie m actually has)
    score = alpha * z_whole + (1 - alpha) * softmin_T(z_clause_i)  - lam * max(0, max_j z_neg_j - tau)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FIELDS, SOURCES, SearchConfig
from .index import MovieIndex
from .lexicon import NEG_META_MAP


@dataclass
class Prepared:
    """Raw (uncorrected) cosines of every query text against every movie and source."""
    texts: list[str]
    n_clauses: int
    n_neg: int
    raw: dict[str, np.ndarray]                      # source -> (n_texts, n_movies), NaN where the source is missing
    whole_chunk_vec: np.ndarray | None = None       # for best-chunk snippets
    extras: dict = field(default_factory=dict)


def raw_scores(index: MovieIndex, field_q: np.ndarray, chunk_q: np.ndarray) -> dict[str, np.ndarray]:
    n_t, n = field_q.shape[0], index.n_movies
    raw: dict[str, np.ndarray] = {}
    for f in FIELDS:
        s = (index.field_emb[f].astype(np.float32) @ field_q.T).T            # (n_t, n)
        s[:, ~index.field_mask[f]] = np.nan
        raw[f] = s
    sub = np.full((n_t, n), np.nan, dtype=np.float32)
    for sh in index.shards:
        if sh.emb.shape[0] == 0:
            continue
        agg = sh.topk_mean(sh.raw_scores(chunk_q), index.cfg.agg_topk)       # (movies_in_shard, n_t)
        sub[:, sh.m0:sh.m0 + sh.n_movies] = agg.T
    raw["subtitles"] = sub
    return raw


def _z(scores: np.ndarray) -> np.ndarray:
    """Row-wise z-score ignoring NaN."""
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    ok = (~np.isnan(scores)).sum(axis=1) > 1
    if ok.any():
        sub = scores[ok]
        mu = np.nanmean(sub, axis=1, keepdims=True)
        sd = np.nanstd(sub, axis=1, keepdims=True)
        out[ok] = (sub - mu) / np.where(sd < 1e-6, 1.0, sd)
    return out


def combine(raw: dict[str, np.ndarray], hub: dict[str, np.ndarray], cfg: SearchConfig, *,
            zscore: bool = True) -> np.ndarray:
    """Weighted fusion over the sources each movie has -> (n_texts, n_movies); NaN if a movie has no source."""
    num = den = None
    for src in SOURCES:
        w = cfg.weights.get(src, 0.0)
        if w <= 0 or src not in raw:
            continue
        s = raw[src]
        if zscore:
            if src in cfg.hub_fields and src in hub:
                s = s - cfg.hub_gamma * hub[src][None, :]
            s = _z(s)
        valid = ~np.isnan(s)
        part = np.where(valid, s, 0.0) * w
        wv = valid * w
        num = part if num is None else num + part
        den = wv if den is None else den + wv
    if num is None:
        raise ValueError("No source with positive weight.")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out.astype(np.float32)


def softmin(z: np.ndarray, T: float) -> np.ndarray:
    """z (n_clauses, n_movies) -> smooth minimum over clauses, NaN-aware."""
    valid = ~np.isnan(z)
    e = np.where(valid, np.exp(-np.nan_to_num(z) / T), 0.0)
    cnt = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, -T * np.log(np.maximum(e.sum(axis=0) / np.maximum(cnt, 1), 1e-30)), np.nan)
    return m.astype(np.float32)


class MetaNegation:
    """D8: negated words that name a genre/keyword ("scary" -> Horror) down-weight movies carrying it."""

    def __init__(self, movies: pd.DataFrame):
        self.genres = movies["genres"].fillna("").str.lower().to_numpy()
        self.keywords = movies["keywords"].fillna("").str.lower().to_numpy()
        self._cache: dict[str, np.ndarray] = {}

    def mask(self, term: str) -> np.ndarray:
        if term in self._cache:
            return self._cache[term]
        m = np.zeros(len(self.genres), dtype=bool)
        for key in {term, *term.split()}:
            spec = NEG_META_MAP.get(key)
            if not spec:
                continue
            for g in spec["genres"]:
                m |= np.fromiter((g.lower() in x.split("|") for x in self.genres), bool, len(self.genres))
            for frag in spec["keywords"]:
                m |= np.fromiter((frag in x for x in self.keywords), bool, len(self.keywords))
        self._cache[term] = m
        return m


def fuse(prep: Prepared, index: MovieIndex, cfg: SearchConfig, *, neg_terms: list[str] | None = None,
         meta: MetaNegation | None = None) -> dict[str, np.ndarray | float]:
    """Stage B: cheap, depends only on the cached raw scores and the SearchConfig."""
    cfg.validate()
    Z = combine(prep.raw, index.hub, cfg)                  # (n_texts, n_movies)
    n_c, n_n = prep.n_clauses, prep.n_neg
    z_whole = Z[0]
    score = z_whole.copy()
    if cfg.use_clauses and n_c >= 2:
        score = cfg.alpha * z_whole + (1 - cfg.alpha) * softmin(Z[1:1 + n_c], cfg.softmin_T)
        score = np.where(np.isnan(score), z_whole, score)
    penalty = np.zeros_like(score)
    if n_n:
        zn = np.nanmax(Z[1 + n_c:1 + n_c + n_n], axis=0)
        penalty = cfg.lam * np.maximum(0.0, np.nan_to_num(zn, nan=-np.inf) - cfg.tau)
        penalty = np.where(np.isfinite(penalty), penalty, 0.0)
        if cfg.meta_negation and meta is not None and neg_terms:
            for t in neg_terms:
                penalty = penalty + cfg.meta_neg_penalty * meta.mask(t)
        score = score - penalty
    raw_whole = combine({k: v[:1] for k, v in prep.raw.items()}, index.hub, cfg, zscore=False)[0]
    confidence = float(np.nanmax(raw_whole)) if np.isfinite(raw_whole).any() else 0.0
    return {"score": score, "z_whole": z_whole, "penalty": penalty, "confidence": confidence}
