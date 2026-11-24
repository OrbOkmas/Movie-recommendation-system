"""Semantic (dense-embedding) movie search — the second approach next to the TF-IDF baseline.

One index stores two kinds of vectors, so representations can be compared without re-encoding:

* **meta**      — one vector per movie: "Title | Genres | Keywords | Plot" (the part of
                  ``text_embedding`` before ``| Dialogue:``).
* **subtitles** — the dialogue is split into word windows ("chunks"); every chunk gets a vector.
                  A movie's subtitle score aggregates its chunk scores (max / mean of top-k / mean).

At search time you choose the *source*:
    meta       -> cosine(query, meta vector)
    subtitles  -> aggregated cosine(query, chunks)
    combined   -> w * meta + (1 - w) * subtitles     (late fusion, w = --meta-weight)

Unlike the TF-IDF baseline, the query is NOT stop-word filtered or lemmatized: the encoder sees the
raw sentence, so negation ("not scary") and misspellings reach the model (whether it handles them
is an empirical question for the error analysis).

From the repository root:
    python -m src.semantic train --data data/processed/sample_clean_movies.csv --output models/semantic_sample
    python -m src.semantic search --index models/semantic_sample --query "dark but not scary" --source combined
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.baseline import BaselineError, load_dataset

LOGGER = logging.getLogger("semantic")
INDEX_VERSION = 1
SOURCES = ("meta", "subtitles", "combined")
AGGREGATIONS = ("max", "topk_mean", "mean")
DIALOGUE_MARKER = "| Dialogue:"
DEBUG_MODEL = "debug-hashing"  # offline stand-in used only to test the pipeline, never for results


class SemanticError(BaselineError):
    """Actionable input, training, or index error for semantic search."""


@dataclass
class SemanticConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_column: str = "text_embedding"
    chunk_words: int = 150          # ~200 word-pieces: fits MiniLM's 256-token window
    chunk_overlap: int = 0
    max_chunks_per_movie: int = 0   # 0 = keep all; >0 = evenly spaced subset (speed/RAM trade-off)
    query_prefix: str = ""          # e.g. "Represent this sentence for searching relevant passages: " for BGE
    doc_prefix: str = ""
    batch_size: int = 64
    query_mode: str = "raw"         # kept for compatibility with the evaluation notebook

    def validate(self) -> None:
        if self.chunk_words < 10:
            raise SemanticError("chunk_words must be at least 10.")
        if not 0 <= self.chunk_overlap < self.chunk_words:
            raise SemanticError("chunk_overlap must be >= 0 and smaller than chunk_words.")
        if self.max_chunks_per_movie < 0:
            raise SemanticError("max_chunks_per_movie must be >= 0.")


# --------------------------------------------------------------------------- text preparation

def split_document(text: str) -> tuple[str, str]:
    """Split ``text_embedding`` into (metadata+plot, dialogue)."""
    text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    if DIALOGUE_MARKER in text:
        head, dialogue = text.split(DIALOGUE_MARKER, 1)
        return head.strip(" |"), dialogue.strip()
    return text.strip(" |"), ""


def chunk_words(text: str, size: int, overlap: int = 0, max_chunks: int = 0) -> list[str]:
    """Word windows of ``size`` words. With ``max_chunks`` keep an evenly spaced subset."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    spans = [[i, min(i + size, len(words))] for i in range(0, max(len(words) - overlap, 1), step)]
    # A tiny trailing window (a few words) can get a spuriously high cosine score under
    # max-aggregation, so a tail shorter than a third of a window is merged into the previous one.
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < size // 3:
        tail_end = spans.pop()[1]
        spans[-1][1] = tail_end
    chunks = [" ".join(words[a:b]) for a, b in spans]
    if max_chunks and len(chunks) > max_chunks:
        keep = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in sorted(set(keep))]
    return chunks


# --------------------------------------------------------------------------- encoders

class _DebugEncoder:
    """Hashing bag-of-words vectors. ONLY for offline smoke tests of the pipeline."""

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def encode(self, texts, batch_size=64, **_):
        return self.vec.transform(texts).toarray().astype(np.float32)


def load_encoder(model_name: str):
    if model_name == DEBUG_MODEL:
        LOGGER.warning("Using the debug hashing encoder: results are NOT semantic, pipeline test only.")
        return _DebugEncoder()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SemanticError("Install sentence-transformers: pip install sentence-transformers") from exc
    return SentenceTransformer(model_name)


def _encode(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    emb = encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True) \
        if not isinstance(encoder, _DebugEncoder) else encoder.encode(texts)
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1, norms)


# --------------------------------------------------------------------------- recommender

class SemanticMovieRecommender:
    def __init__(self, config: SemanticConfig | None = None):
        self.config = config or SemanticConfig()
        self.config.validate()
        self.meta_emb: np.ndarray | None = None      # (n_movies, d) float16
        self.chunk_emb: np.ndarray | None = None     # (n_chunks, d) float16
        self.chunk_movie: np.ndarray | None = None   # (n_chunks,) int32, sorted by movie
        self.chunk_text: list[str] | None = None
        self.metadata: pd.DataFrame | None = None
        self.stats: dict[str, Any] = {}
        self._encoder = None

    # ---------------------------------------------------------------- training
    def fit(self, df: pd.DataFrame, *, store_chunk_text: bool = True,
            encode_buffer: int = 4096) -> "SemanticMovieRecommender":
        c = self.config
        for col in ("imdb_id", "tmdb_title", c.text_column):
            if col not in df.columns:
                raise SemanticError(f"Cannot fit: missing column {col!r}.")
        work = df.copy()
        work["imdb_id"] = work["imdb_id"].astype(str).str.strip()
        work = work[work["imdb_id"].ne("") & work[c.text_column].notna()]
        dup = work["imdb_id"].duplicated()
        if dup.any():
            LOGGER.warning("Dropping %d duplicate imdb_id rows.", int(dup.sum()))
            work = work[~dup]
        work = work.reset_index(drop=True)
        if work.empty:
            raise SemanticError("No valid rows to index.")

        self._encoder = self._encoder or load_encoder(c.model_name)
        t0 = time.time()

        heads, n_without_dialogue = [], 0
        chunk_vecs, chunk_movie, chunk_text = [], [], []
        buf_texts, buf_movie = [], []
        n_chunks_total, n_chunks_before_cap = 0, 0

        def flush():
            if not buf_texts:
                return
            chunk_vecs.append(_encode(self._encoder, [c.doc_prefix + t for t in buf_texts],
                                      c.batch_size).astype(np.float16))
            chunk_movie.extend(buf_movie)
            if store_chunk_text:
                chunk_text.extend(buf_texts)
            buf_texts.clear(); buf_movie.clear()

        for i, text in enumerate(work[c.text_column]):
            head, dialogue = split_document(text)
            heads.append(head)
            all_chunks = chunk_words(dialogue, c.chunk_words, c.chunk_overlap)
            n_chunks_before_cap += len(all_chunks)
            chunks = chunk_words(dialogue, c.chunk_words, c.chunk_overlap, c.max_chunks_per_movie)
            if not chunks:
                n_without_dialogue += 1
            buf_texts.extend(chunks); buf_movie.extend([i] * len(chunks))
            n_chunks_total += len(chunks)
            if len(buf_texts) >= encode_buffer:
                flush()
                LOGGER.info("  encoded %d movies / %d chunks (%.0fs)", i + 1, n_chunks_total, time.time() - t0)
        flush()

        LOGGER.info("Encoding %d metadata+plot texts ...", len(heads))
        meta = []
        for s in range(0, len(heads), encode_buffer):
            meta.append(_encode(self._encoder, [c.doc_prefix + h for h in heads[s:s + encode_buffer]],
                                c.batch_size).astype(np.float16))

        self.meta_emb = np.vstack(meta)
        self.chunk_emb = np.vstack(chunk_vecs) if chunk_vecs else np.zeros((0, self.meta_emb.shape[1]), np.float16)
        self.chunk_movie = np.asarray(chunk_movie, dtype=np.int32)
        self.chunk_text = chunk_text if store_chunk_text else None
        keep = [x for x in ("imdb_id", "tmdb_title", "release_year", "genres") if x in work.columns]
        self.metadata = work[keep].copy()
        self.stats = {
            "indexed_movies": len(work),
            "movies_without_dialogue": n_without_dialogue,
            "n_chunks": int(n_chunks_total),
            "n_chunks_before_cap": int(n_chunks_before_cap),
            "mean_chunks_per_movie": round(n_chunks_total / len(work), 2),
            "embedding_dim": int(self.meta_emb.shape[1]),
            "encode_seconds": round(time.time() - t0, 1),
        }
        LOGGER.info("Fitted semantic index: %s", self.stats)
        return self

    # ---------------------------------------------------------------- scoring
    def _query_vector(self, query: str) -> np.ndarray:
        self._encoder = self._encoder or load_encoder(self.config.model_name)
        return _encode(self._encoder, [self.config.query_prefix + query], 1)[0]

    @staticmethod
    def _dot(matrix16: np.ndarray, q: np.ndarray, block: int = 200_000) -> np.ndarray:
        out = np.empty(matrix16.shape[0], dtype=np.float32)
        for s in range(0, matrix16.shape[0], block):
            out[s:s + block] = matrix16[s:s + block].astype(np.float32) @ q
        return out

    def _aggregate(self, chunk_scores: np.ndarray, how: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-movie subtitle score + index of the best chunk (for explanations)."""
        n = len(self.metadata)
        score = np.full(n, np.nan, dtype=np.float32)
        best = np.full(n, -1, dtype=np.int64)
        if len(chunk_scores) == 0:
            return score, best
        movie = self.chunk_movie
        order = np.lexsort((-chunk_scores, movie))            # by movie, then descending score
        m_sorted, s_sorted = movie[order], chunk_scores[order]
        starts = np.flatnonzero(np.r_[True, m_sorted[1:] != m_sorted[:-1]])
        movies_present = m_sorted[starts]
        best[movies_present] = order[starts]
        if how == "max":
            score[movies_present] = s_sorted[starts]
        else:
            rank = np.arange(len(m_sorted)) - np.repeat(starts, np.diff(np.r_[starts, len(m_sorted)]))
            mask = rank < k if how == "topk_mean" else np.ones_like(rank, dtype=bool)
            sums = np.bincount(m_sorted[mask], weights=s_sorted[mask], minlength=n)
            cnts = np.bincount(m_sorted[mask], minlength=n)
            score[movies_present] = (sums[movies_present] / cnts[movies_present]).astype(np.float32)
        return score, best

    def search(self, query: str, *, top_k: int = 5, source: str = "combined",
               aggregation: str = "topk_mean", agg_k: int = 3, meta_weight: float = 0.5,
               exclude_imdb_id: str | None = None, snippet_words: int = 40) -> list[dict[str, Any]]:
        if self.meta_emb is None or self.metadata is None:
            raise SemanticError("Model not fitted. Train or load an index first.")
        if source not in SOURCES:
            raise SemanticError(f"source must be one of {SOURCES}.")
        if aggregation not in AGGREGATIONS:
            raise SemanticError(f"aggregation must be one of {AGGREGATIONS}.")
        if not isinstance(top_k, int) or top_k < 1:
            raise SemanticError("top_k must be a positive integer.")
        if not isinstance(query, str) or not query.strip():
            LOGGER.warning("Empty query: nothing to search for.")
            return []

        q = self._query_vector(query.strip())
        meta_s = self._dot(self.meta_emb, q)
        sub_s, best_chunk = self._aggregate(self._dot(self.chunk_emb, q), aggregation, agg_k)

        if source == "meta":
            final = meta_s.copy()
        elif source == "subtitles":
            final = sub_s.copy()                                   # NaN for movies without dialogue
        else:
            final = np.where(np.isnan(sub_s), meta_s, meta_weight * meta_s + (1 - meta_weight) * sub_s)

        valid = ~np.isnan(final)
        if exclude_imdb_id is not None:
            valid &= self.metadata["imdb_id"].astype(str).to_numpy() != str(exclude_imdb_id)
        idx = np.flatnonzero(valid)
        idx = idx[np.lexsort((idx, -final[idx]))][:top_k]

        found = []
        for rank, pos in enumerate(idx, start=1):
            item = self.metadata.iloc[int(pos)]
            movie: dict[str, Any] = {
                "rank": rank, "movie_id": str(item["imdb_id"]), "title": str(item["tmdb_title"]),
                "score": round(float(final[pos]), 6),
                "meta_score": round(float(meta_s[pos]), 6),
                "subtitle_score": None if np.isnan(sub_s[pos]) else round(float(sub_s[pos]), 6),
            }
            for col in ("release_year", "genres"):
                if col in self.metadata.columns and pd.notna(item[col]):
                    movie[col] = str(item[col])
            if self.chunk_text is not None and best_chunk[pos] >= 0 and source != "meta":
                words = self.chunk_text[int(best_chunk[pos])].split()
                movie["best_chunk"] = " ".join(words[:snippet_words]) + (" ..." if len(words) > snippet_words else "")
            found.append(movie)
        return found

    # ---------------------------------------------------------------- persistence
    def save(self, index_dir: str | Path, *, overwrite: bool = False) -> Path:
        if self.meta_emb is None:
            raise SemanticError("Nothing to save; fit a model first.")
        index_dir = Path(index_dir)
        if index_dir.exists() and any(index_dir.iterdir()) and not overwrite:
            raise SemanticError(f"Index directory {index_dir} isn't empty. Pass --overwrite to update it.")
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "meta_embeddings.npy", self.meta_emb)
        np.save(index_dir / "chunk_embeddings.npy", self.chunk_emb)
        np.save(index_dir / "chunk_movie.npy", self.chunk_movie)
        if self.chunk_text is not None:
            pd.DataFrame({"chunk_text": self.chunk_text}).to_parquet(index_dir / "chunk_text.parquet", index=False)
        self.metadata.to_csv(index_dir / "movie_metadata.csv", index=False, encoding="utf-8")
        (index_dir / "manifest.json").write_text(json.dumps(
            {"index_version": INDEX_VERSION, "method": "semantic", "config": asdict(self.config), "stats": self.stats},
            indent=2, ensure_ascii=False), encoding="utf-8")
        return index_dir

    @classmethod
    def load(cls, index_dir: str | Path, *, load_chunk_text: bool = True) -> "SemanticMovieRecommender":
        index_dir = Path(index_dir)
        need = ("manifest.json", "meta_embeddings.npy", "chunk_embeddings.npy", "chunk_movie.npy", "movie_metadata.csv")
        absent = [f for f in need if not (index_dir / f).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing semantic index files in {index_dir}: {absent}")
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("index_version") != INDEX_VERSION:
            raise SemanticError("Incompatible index version. Retrain this index.")
        model = cls(SemanticConfig(**manifest["config"]))
        model.meta_emb = np.load(index_dir / "meta_embeddings.npy")
        model.chunk_emb = np.load(index_dir / "chunk_embeddings.npy", mmap_mode="r")
        model.chunk_movie = np.load(index_dir / "chunk_movie.npy")
        model.metadata = pd.read_csv(index_dir / "movie_metadata.csv", dtype={"imdb_id": "string"})
        if load_chunk_text and (index_dir / "chunk_text.parquet").is_file():
            model.chunk_text = pd.read_parquet(index_dir / "chunk_text.parquet")["chunk_text"].tolist()
        model.stats = manifest.get("stats", {})
        if len(model.meta_emb) != len(model.metadata) or len(model.chunk_emb) != len(model.chunk_movie):
            raise SemanticError("Corrupt index: embedding and metadata counts don't match.")
        return model


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic (embedding) movie search")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Encode movies and save an index")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--model", default=SemanticConfig.model_name)
    t.add_argument("--text-column", default="text_embedding")
    t.add_argument("--csv-separator", default=";")
    t.add_argument("--max-rows", type=int)
    t.add_argument("--chunk-words", type=int, default=150)
    t.add_argument("--chunk-overlap", type=int, default=0)
    t.add_argument("--max-chunks-per-movie", type=int, default=0)
    t.add_argument("--query-prefix", default="")
    t.add_argument("--doc-prefix", default="")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--no-chunk-text", action="store_true", help="Do not store chunk texts (saves disk; no explanations)")
    t.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("search", help="Query a saved index")
    s.add_argument("--index", type=Path, required=True)
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--source", choices=SOURCES, default="combined")
    s.add_argument("--aggregation", choices=AGGREGATIONS, default="topk_mean")
    s.add_argument("--agg-k", type=int, default=3)
    s.add_argument("--meta-weight", type=float, default=0.5)
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        if args.command == "train":
            cfg = SemanticConfig(model_name=args.model, text_column=args.text_column,
                                 chunk_words=args.chunk_words, chunk_overlap=args.chunk_overlap,
                                 max_chunks_per_movie=args.max_chunks_per_movie,
                                 query_prefix=args.query_prefix, doc_prefix=args.doc_prefix,
                                 batch_size=args.batch_size)
            df = load_dataset(args.data, args.text_column, max_rows=args.max_rows, csv_separator=args.csv_separator)
            model = SemanticMovieRecommender(cfg).fit(df, store_chunk_text=not args.no_chunk_text)
            print("Saved index to", model.save(args.output, overwrite=args.overwrite))
            print(json.dumps(model.stats, indent=2))
        else:
            model = SemanticMovieRecommender.load(args.index)
            res = model.search(args.query, top_k=args.top_k, source=args.source,
                               aggregation=args.aggregation, agg_k=args.agg_k, meta_weight=args.meta_weight)
            print(json.dumps(res, indent=2, ensure_ascii=False) if res else "No results.")
    except (BaselineError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
