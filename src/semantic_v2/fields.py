"""Turn the cleaned movie table into per-field texts (plot, keywords, genres) and dialogue chunks.

Rules (see docs/semantic_v2_plan.md, D1-D3):
* title is NOT embedded; it is only kept for display
* a missing field is simply missing (no vector, no score) - never an empty-string embedding
* a movie with no plot, no keywords and no usable dialogue is not rankable and is skipped
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import BuildConfig, SemanticV2Error

LOGGER = logging.getLogger("semantic_v2")
DIALOGUE_MARKER = "| Dialogue:"
NEEDED_COLUMNS = ("imdb_id", "tmdb_title", "release_year", "genres", "keywords", "clean_plot", "text_embedding")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def is_missing(x) -> bool:
    if x is None or x is pd.NA:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return isinstance(x, str) and not x.strip()


def split_document(text) -> tuple[str, str]:
    """Split ``text_embedding`` into (metadata+plot head, dialogue)."""
    if is_missing(text):
        return "", ""
    text = str(text)
    if DIALOGUE_MARKER in text:
        head, dialogue = text.split(DIALOGUE_MARKER, 1)
        return head.strip(" |"), dialogue.strip()
    return text.strip(" |"), ""


def plot_from_head(head: str) -> str:
    m = re.search(r"\|\s*Plot:\s*(.*)$", head, flags=re.S)
    return m.group(1).strip() if m else ""


def chunk_words(text: str, size: int, overlap: int = 0, max_chunks: int = 0) -> list[str]:
    """Word windows; a tail shorter than a third of a window is merged into the previous window.

    (Same behaviour as v1 after its off-by-one fix; re-implemented here so v2 has no dependency on v1.)
    """
    words = text.split()
    if not words:
        return []
    step = size - overlap
    spans = [[i, min(i + size, len(words))] for i in range(0, max(len(words) - overlap, 1), step)]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < size // 3:
        tail_end = spans.pop()[1]
        spans[-1][1] = tail_end
    chunks = [" ".join(words[a:b]) for a, b in spans]
    if max_chunks and len(chunks) > max_chunks:
        keep = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in sorted(set(keep))]
    return chunks


def clean_plot_text(x, max_words: int) -> str:
    if is_missing(x):
        return ""
    return " ".join(str(x).split()[:max_words])


def split_pipe(x) -> list[str]:
    if is_missing(x):
        return []
    seen, out = set(), []
    for part in str(x).split("|"):
        part = part.strip()
        if part and part.lower() not in seen:
            seen.add(part.lower())
            out.append(part)
    return out


def keywords_text(x, max_keywords: int, template: str) -> str:
    kws = split_pipe(x)[:max_keywords]
    return template.format(", ".join(kws)) if kws else ""


def genres_text(x, template: str) -> str:
    gs = split_pipe(x)
    return template.format(", ".join(gs)) if gs else ""


@dataclass
class MovieRecord:
    imdb_id: str
    title: str
    year: str
    genres_raw: str
    keywords_raw: str
    plot: str
    plot_text: str
    keywords_text: str
    genres_text: str
    chunks: list[str] = field(default_factory=list)


@dataclass
class LoadStats:
    rows_read: int = 0
    duplicate_ids: int = 0
    excluded_no_text: int = 0
    kept: int = 0
    no_plot: int = 0
    no_keywords: int = 0
    no_genres: int = 0
    no_dialogue: int = 0


def _year(x) -> str:
    if is_missing(x):
        return ""
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x)


def _detect_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    return ";" if header.count(";") > header.count(",") else ","


def iter_frames(path: str | Path, *, batch_rows: int = 2000, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    """Stream the dataset in batches, selecting only the needed columns (RAM-safe for the full parquet)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    left = max_rows
    suffix = path.suffix.lower()
    if suffix == ".csv":
        sep = _detect_sep(path)
        reader = pd.read_csv(path, sep=sep, encoding="utf-8-sig", usecols=lambda c: c in NEEDED_COLUMNS,
                             dtype=str, chunksize=batch_rows, low_memory=False)
        for df in reader:
            if left is not None:
                df = df.head(left)
                left -= len(df)
            yield df
            if left is not None and left <= 0:
                return
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SemanticV2Error("Reading parquet needs pyarrow: pip install pyarrow") from exc
        pf = pq.ParquetFile(path)
        cols = [c for c in NEEDED_COLUMNS if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            df = batch.to_pandas().astype(object)
            if left is not None:
                df = df.head(left)
                left -= len(df)
            yield df
            if left is not None and left <= 0:
                return
    else:
        raise SemanticV2Error(f"Unsupported file type {suffix!r}; use .csv or .parquet.")


def iter_movies(path: str | Path, cfg: BuildConfig, stats: LoadStats | None = None, *,
                max_rows: int | None = None) -> Iterator[MovieRecord]:
    """Yield rankable movies in file order (deterministic: required for resumable shards)."""
    stats = stats if stats is not None else LoadStats()
    seen: set[str] = set()
    for df in iter_frames(path, max_rows=max_rows):
        for col in ("imdb_id", "tmdb_title"):
            if col not in df.columns:
                raise SemanticV2Error(f"Missing required column {col!r}; available: {list(df.columns)}")
        for row in df.to_dict("records"):
            stats.rows_read += 1
            imdb = str(row.get("imdb_id", "")).strip()
            if not imdb or imdb.lower() == "nan":
                continue
            if imdb in seen:
                stats.duplicate_ids += 1
                continue
            seen.add(imdb)
            head, dialogue = split_document(row.get("text_embedding"))
            plot = clean_plot_text(row.get("clean_plot"), cfg.max_plot_words) or clean_plot_text(plot_from_head(head), cfg.max_plot_words)
            kw = keywords_text(row.get("keywords"), cfg.max_keywords, cfg.keywords_template)
            gn = genres_text(row.get("genres"), cfg.genres_template)
            chunks = []
            if len(dialogue.split()) >= cfg.min_dialogue_words:
                chunks = chunk_words(dialogue, cfg.chunk_words, cfg.chunk_overlap, cfg.max_chunks_per_movie)
            if not (plot or kw or chunks):
                stats.excluded_no_text += 1
                continue
            stats.kept += 1
            stats.no_plot += not plot
            stats.no_keywords += not kw
            stats.no_genres += not gn
            stats.no_dialogue += not chunks
            year = _year(row.get("release_year"))
            yield MovieRecord(imdb_id=imdb, title=str(row["tmdb_title"]), year=year,
                              genres_raw="" if is_missing(row.get("genres")) else str(row["genres"]),
                              keywords_raw="" if is_missing(row.get("keywords")) else str(row["keywords"]),
                              plot=plot, plot_text=plot, keywords_text=kw, genres_text=gn, chunks=chunks)


def vocab_counts(rec: MovieRecord, counter: dict[str, int]) -> None:
    """Word counts over plot/keywords/genres (for the spell checker's domain vocabulary)."""
    for text in (rec.plot, rec.keywords_raw.replace("|", " "), rec.genres_raw.replace("|", " "), rec.title):
        for w in _WORD.findall(text.lower()):
            counter[w] = counter.get(w, 0) + 1
