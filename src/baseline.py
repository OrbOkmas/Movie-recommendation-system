"""A reusable TF-IDF movie recommendation baseline.

Designed for  ``sample_clean_movies.csv`` and ``clean_movies.parquet``.
Fits a sparse TF-IDF index once, saves it, and ranks with a sparse query dot
product (never constructs an all-movies-by-all-movies similarity matrix).

From the repository root:
    python -m src.baseline train --data data/processed/sample_clean_movies.csv --output models/tfidf_sample
    python -m src.baseline search --index models/tfidf_sample --query 'a tense mystery in a remote town'
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

LOGGER = logging.getLogger("baseline")
REQUIRED = ("imdb_id", "tmdb_title")
OPTIONAL_META = ("release_year", "genres")
INDEX_VERSION = 1


class BaselineError(ValueError):
    """Actionable input, training, or index error."""


def load_dataset(path: str | Path, text_column: str = "text_tfidf", *,
                 max_rows: int | None = None, csv_separator: str = ";") -> pd.DataFrame:
    """Load necessary fields, keeping movie IDs as strings.

    Parquet uses PyArrow for column selection and optional row-limited iteration.
    Full Parquet still needs enough RAM to hold the selected text columns.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Check the path from the project root."
        )
    if not text_column or text_column in REQUIRED:
        raise BaselineError("--text-column must name a text field, e.g. text_tfidf or clean_plot.")
    if max_rows is not None and max_rows < 1:
        raise BaselineError("--max-rows must be a positive integer.")

    requested = set(REQUIRED) | {text_column} | set(OPTIONAL_META)
    if path.suffix.lower() == ".csv":
        # A callable selects only useful columns even in a very wide CSV.
        try:
            data = pd.read_csv(
                path, sep=csv_separator, encoding="utf-8-sig",
                usecols=lambda col: col in requested,
                dtype={"imdb_id": "string"}, nrows=max_rows,
                low_memory=False,
            )
        except (pd.errors.ParserError, UnicodeError) as exc:
            raise BaselineError(f"Could not parse CSV {path}: {exc}") from exc
    elif path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise BaselineError(
                "Parquet reading requires PyArrow: pip install pyarrow"
            ) from exc
        try:
            parquet = pq.ParquetFile(path)
            columns = [x for x in requested if x in parquet.schema_arrow.names]
            if max_rows is None:
                data = parquet.read(columns=columns).to_pandas()
            else:
                frames = []
                left = max_rows
                for batch in parquet.iter_batches(batch_size=min(1024, left), columns=columns):
                    frames.append(batch.to_pandas())
                    left -= len(batch)
                    if left <= 0:
                        break
                data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        except Exception as exc:
            raise BaselineError(f"Could not read Parquet {path}: {exc}") from exc
    else:
        raise BaselineError(f"Unsupported file type {path.suffix!r}; use .csv or .parquet.")

    missing = [c for c in (*REQUIRED, text_column) if c not in data.columns]
    if missing:
        raise BaselineError(
            f"Missing required columns {missing}; available: {list(data.columns)}. "
            f"For the uploaded sample, use --text-column text_tfidf."
        )
    data["imdb_id"] = data["imdb_id"].astype("string")
    return data


def _normalize_query(query: str, mode: str) -> str:
    """Prepare raw queries for Vojta's already-lemmatized text_tfidf column.

    'wordnet' mirrors preprocessing.py's English spaCy tokenization,
    stop-word removal and WordNet noun lemmatization, and strips initial filler.
    'basic' remains available as a dependency-light diagnostic alternative.
    """
    if not isinstance(query, str) or not query.strip():
        raise BaselineError("Query must be a nonempty string.")
    if mode not in {"basic", "wordnet"}:
        raise BaselineError(f"Unknown query normalization mode: {mode}")

    from src.tfidf_text import strip_query_filler, preprocess_tfidf_query
    if mode == "wordnet":
        try:
            return preprocess_tfidf_query(query)
        except (RuntimeError, ImportError) as exc:
            raise BaselineError(str(exc)) from exc
    # Legacy basic mode, available for diagnostic comparison only.
    cleaned_query = strip_query_filler(query)
    words = re.findall(r"[^\W\d_]+", cleaned_query.lower(), flags=re.UNICODE)
    return " ".join(w for w in words if len(w) > 1 and w not in ENGLISH_STOP_WORDS)


@dataclass
class BaselineConfig:
    text_column: str = "text_tfidf"
    max_features: int = 50_000
    max_words_per_doc: int = 4_000
    ngram_max: int = 2
    min_df: int = 1
    duplicate_policy: str = "error"  # 'error' or 'first'
    query_mode: str = "wordnet"  # aligned to Vojta's already-normalized text_tfidf
    max_total_chars: int = 350_000_000  # pre-fit memory safeguard; override consciously

    def validate(self) -> None:
        if self.max_features < 100 or self.ngram_max not in (1, 2, 3):
            raise BaselineError("max_features must be >=100 and ngram_max must be 1, 2 or 3.")
        if self.max_words_per_doc < 1 or self.min_df < 1:
            raise BaselineError("max_words_per_doc and min_df must be positive.")
        if self.duplicate_policy not in {"error", "first"}:
            raise BaselineError("duplicate_policy must be 'error' or 'first'.")
        if self.query_mode not in {"basic", "wordnet"}:
            raise BaselineError("query_mode must be 'basic' or 'wordnet'.")
        if self.max_total_chars < 1:
            raise BaselineError("max_total_chars must be positive.")


class TfidfMovieRecommender:
    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig()
        self.config.validate()
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: sparse.csr_matrix | None = None
        self.metadata: pd.DataFrame | None = None
        self.stats: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame, *, allow_large: bool = False) -> "TfidfMovieRecommender":
        required = [*REQUIRED, self.config.text_column]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise BaselineError(f"Cannot fit: missing columns {missing}.")
        if df.empty:
            raise BaselineError("Cannot fit an empty dataset.")

        source_rows = len(df)
        keep_cols = required + [c for c in OPTIONAL_META if c in df.columns]
        work = df.loc[:, keep_cols].copy()
        for c in required:
            work[c] = work[c].fillna("").astype(str).str.strip()
        nonempty = work[required].ne("").all(axis=1)
        dropped = int((~nonempty).sum())
        if dropped:
            LOGGER.warning("Ignoring %d records with missing IDs, titles or text.", dropped)
        work = work.loc[nonempty].reset_index(drop=True)
        if work.empty:
            raise BaselineError("No valid records remain (check IDs, titles and text).")
        duplicates = work["imdb_id"].duplicated(keep="first")
        if duplicates.any():
            examples = work.loc[duplicates, "imdb_id"].head(3).tolist()
            if self.config.duplicate_policy == "error":
                raise BaselineError(
                    f"Found {int(duplicates.sum())} duplicate imdb_id values (e.g. {examples}). "
                    "Fix the data, or pass --duplicates first to keep the first record."
                )
            LOGGER.warning("Dropping %d duplicate movie IDs.", int(duplicates.sum()))
            work = work.loc[~duplicates].reset_index(drop=True)
        # Cap documents before measuring the text that the vectorizer will process.
        # This guard limits TF-IDF fitting work, not the RAM already used by loading
        # a large input file; see BASELINE_SETUP.md for full-data precautions.
        texts: list[str] = []
        truncated = 0
        cap = self.config.max_words_per_doc
        for text in work[self.config.text_column]:
            words = text.split()
            if len(words) > cap:
                truncated += 1
                words = words[:cap]
            texts.append(" ".join(words))
        if truncated:
            LOGGER.warning(
                "Truncated %d/%d documents to %d words. This may alter subtitle relevance; "
                "record this setting in evaluation.", truncated, len(work), cap
            )
        total_chars = sum(map(len, texts))
        if total_chars > self.config.max_total_chars and not allow_large:
            raise BaselineError(
                f"Input text after document caps totals ~{total_chars / 1e6:.0f} million characters. "
                "Train on fewer records or lower max_words_per_doc, or pass --allow-large "
                "after checking available RAM."
            )

        vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, self.config.ngram_max),
            max_features=self.config.max_features,
            min_df=self.config.min_df, dtype=np.float32,
            norm="l2", sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(texts).tocsr()
        except ValueError as exc:
            raise BaselineError(
                f"TF-IDF training failed: {exc}. Try more valid documents, min_df=1, "
                "or a less restrictive text representation."
            ) from exc
        if matrix.nnz == 0:
            raise BaselineError("All fitted TF-IDF vectors are empty.")

        self.vectorizer, self.matrix = vectorizer, matrix
        self.metadata = work[[c for c in ["imdb_id", "tmdb_title", *OPTIONAL_META] if c in work.columns]].copy()
        self.stats = {
            "input_rows": source_rows, "indexed_rows": len(work),
            "dropped_invalid_rows": dropped, "truncated_documents": truncated,
            "n_features": matrix.shape[1], "nonzero_elements": int(matrix.nnz),
            "text_column": self.config.text_column,
        }
        LOGGER.info("Fitted TF-IDF baseline: %s", self.stats)
        return self

    def search(self, query: str, *, top_k: int = 5,
               exclude_imdb_id: str | None = None) -> list[dict[str, Any]]:
        if self.vectorizer is None or self.matrix is None or self.metadata is None:
            raise BaselineError("Model not fitted. Train or load an index first.")
        if not isinstance(top_k, int) or top_k < 1:
            raise BaselineError("top_k must be a positive integer.")
        normalized = _normalize_query(query, self.config.query_mode)
        if not normalized:
            LOGGER.warning("The query has no usable words after normalization.")
            return []
        query_vec = self.vectorizer.transform([normalized])
        if query_vec.nnz == 0:
            LOGGER.warning("No words in the query appear in the fitted vocabulary.")
            return []

        # The sparse (movies × terms) @ (terms × 1) result is one small
        # score per movie, not an enormous movies × movies dense matrix.
        scores = (self.matrix @ query_vec.T).toarray().ravel()
        candidate_idx = np.flatnonzero(scores > 0)
        if exclude_imdb_id is not None:
            ids = self.metadata["imdb_id"].to_numpy(dtype=str)
            candidate_idx = candidate_idx[ids[candidate_idx] != str(exclude_imdb_id)]
        if len(candidate_idx) == 0:
            return []
        # Sort by decreasing similarity, tie-break by original index.
        ordering = np.lexsort((candidate_idx, -scores[candidate_idx]))
        top = candidate_idx[ordering[:top_k]]
        found = []
        for rank, pos in enumerate(top, start=1):
            item = self.metadata.iloc[int(pos)]
            movie: dict[str, Any] = {
                "rank": rank,
                "movie_id": str(item["imdb_id"]),
                "title": str(item["tmdb_title"]),
                "score": round(float(scores[pos]), 6),
            }
            for column in OPTIONAL_META:
                if column in self.metadata.columns and pd.notna(item[column]):
                    movie[column] = str(item[column])
            found.append(movie)
        return found

    def save(self, index_dir: str | Path, *, overwrite: bool = False) -> Path:
        if self.vectorizer is None or self.matrix is None or self.metadata is None:
            raise BaselineError("Nothing to save; fit a model first.")
        index_dir = Path(index_dir)
        if index_dir.exists() and any(index_dir.iterdir()) and not overwrite:
            raise BaselineError(
                f"Index directory {index_dir} isn't empty. Pass --overwrite to update it."
            )
        index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, index_dir / "tfidf_vectorizer.joblib")
        sparse.save_npz(index_dir / "tfidf_matrix.npz", self.matrix)
        self.metadata.to_csv(index_dir / "movie_metadata.csv", index=False, encoding="utf-8")
        manifest = {
            "index_version": INDEX_VERSION,
            "config": vars(self.config),
            "stats": self.stats,
        }
        (index_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return index_dir

    @classmethod
    def load(cls, index_dir: str | Path) -> "TfidfMovieRecommender":
        index_dir = Path(index_dir)
        required_files = (
            "manifest.json", "tfidf_vectorizer.joblib", "tfidf_matrix.npz", "movie_metadata.csv"
        )
        absent = [f for f in required_files if not (index_dir / f).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing saved index files in {index_dir}: {absent}")
        try:
            manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("index_version") != INDEX_VERSION:
                raise BaselineError("Incompatible index version. Retrain this index.")
            model = cls(BaselineConfig(**manifest["config"]))
            # Only load joblib files created by your own team: pickle is unsafe for untrusted files.
            model.vectorizer = joblib.load(index_dir / "tfidf_vectorizer.joblib")
            model.matrix = sparse.load_npz(index_dir / "tfidf_matrix.npz").tocsr()
            model.metadata = pd.read_csv(index_dir / "movie_metadata.csv", dtype={"imdb_id": "string"})
            model.stats = manifest["stats"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise BaselineError(f"Could not load saved index: {exc}") from exc
        if model.matrix.shape[0] != len(model.metadata):
            raise BaselineError("Corrupt index: vector and movie metadata counts don't match.")
        if model.matrix.shape[1] != len(model.vectorizer.vocabulary_):
            raise BaselineError("Corrupt index: vector and vocabulary sizes don't match.")
        return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TF-IDF Movie Recommendation Baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Fit model from CSV or Parquet and save index")
    train.add_argument("--data", type=Path, required=True, help="Path to sample CSV or full Parquet")
    train.add_argument("--output", type=Path, required=True, help="Index output directory")
    train.add_argument("--text-column", default="text_tfidf")
    train.add_argument("--csv-separator", default=";")
    train.add_argument("--max-rows", type=int)
    train.add_argument("--max-features", type=int, default=50_000)
    train.add_argument("--max-words-per-doc", type=int, default=4_000)
    train.add_argument("--ngram-max", type=int, default=2)
    train.add_argument("--min-df", type=int, default=1)
    train.add_argument("--duplicates", choices=["error", "first"], default="error")
    train.add_argument("--query-mode", choices=["basic", "wordnet"], default="wordnet")
    train.add_argument("--allow-large", action="store_true", help="Override pre-fit text-size RAM guard")
    train.add_argument("--overwrite", action="store_true", help="Overwrite files in existing index directory")

    search = sub.add_parser("search", help="Rank movies from saved index")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--exclude-imdb-id")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    try:
        if args.command == "train":
            config = BaselineConfig(
                text_column=args.text_column, max_features=args.max_features,
                max_words_per_doc=args.max_words_per_doc,
                ngram_max=args.ngram_max, min_df=args.min_df,
                duplicate_policy=args.duplicates, query_mode=args.query_mode,
            )
            data = load_dataset(
                args.data, args.text_column, max_rows=args.max_rows,
                csv_separator=args.csv_separator,
            )
            model = TfidfMovieRecommender(config).fit(data, allow_large=args.allow_large)
            model.save(args.output, overwrite=args.overwrite)
            print(json.dumps({"index": str(args.output), **model.stats}, indent=2))
        else:
            model = TfidfMovieRecommender.load(args.index)
            results = model.search(args.query, top_k=args.top_k,
                                   exclude_imdb_id=args.exclude_imdb_id)
            print(json.dumps({"query": args.query, "method": "tfidf",
                              "text_source": model.config.text_column,
                              "recommendations": results}, indent=2, ensure_ascii=False))
    except (BaselineError, FileNotFoundError, PermissionError, MemoryError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
