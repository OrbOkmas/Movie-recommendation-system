"""Chunk shards and per-movie aggregation of chunk scores (vectorised top-k mean)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ChunkShard:
    emb: np.ndarray            # (n_chunks, d) float16, sorted by movie
    movie_local: np.ndarray    # (n_chunks,) int32, ascending
    m0: int                    # global index of the first movie of the shard
    n_movies: int
    text_path: Path | None = None
    _layout: tuple | None = field(default=None, repr=False)
    _texts: object = field(default=None, repr=False)

    def layout(self):
        if self._layout is None:
            counts = np.bincount(self.movie_local, minlength=self.n_movies)
            starts = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
            maxc = int(counts.max()) if len(counts) else 0
            pos = np.arange(maxc)[None, :]
            valid = pos < counts[:, None]
            idx = np.where(valid, starts[:, None] + pos, 0).astype(np.int64)
            self._layout = (counts, starts, idx, valid)
        return self._layout

    def raw_scores(self, qmat: np.ndarray, block: int = 200_000) -> np.ndarray:
        """(n_chunks, n_queries) float32 cosine scores; qmat is (n_queries, d) float32."""
        out = np.empty((self.emb.shape[0], qmat.shape[0]), dtype=np.float32)
        qt = qmat.T.astype(np.float32)
        for s in range(0, self.emb.shape[0], block):
            out[s:s + block] = self.emb[s:s + block].astype(np.float32) @ qt
        return out

    def topk_mean(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Per-movie mean of the k best chunk scores -> (n_movies, n_queries); NaN for movies without chunks."""
        counts, _, idx, valid = self.layout()
        n_q = scores.shape[1]
        out = np.full((self.n_movies, n_q), np.nan, dtype=np.float32)
        has = counts > 0
        if not has.any():
            return out
        for j in range(n_q):
            g = scores[idx, j]
            g = np.where(valid, g, -np.inf)
            kk = min(k, g.shape[1])
            top = -np.partition(-g, kk - 1, axis=1)[:, :kk]
            finite = np.isfinite(top)
            sums = np.where(finite, top, 0.0).sum(axis=1)
            cnt = finite.sum(axis=1)
            col = np.full(self.n_movies, np.nan, dtype=np.float32)
            col[has] = sums[has] / np.maximum(cnt[has], 1)
            out[:, j] = col
        return out

    def best_chunk_text(self, movie_local: int, q: np.ndarray, words: int = 40) -> str | None:
        counts, starts, _, _ = self.layout()
        if counts[movie_local] == 0 or self.text_path is None or not self.text_path.is_file():
            return None
        a, b = int(starts[movie_local]), int(starts[movie_local] + counts[movie_local])
        best = a + int(np.argmax(self.emb[a:b].astype(np.float32) @ q))
        if self._texts is None:
            import pandas as pd
            self._texts = pd.read_parquet(self.text_path)["chunk_text"].tolist()
        toks = self._texts[best].split()
        return " ".join(toks[:words]) + (" ..." if len(toks) > words else "")
