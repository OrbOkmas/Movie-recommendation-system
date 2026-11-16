import os
import re
import unicodedata
from typing import Iterator
import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from tqdm import tqdm

# Pre-compiled regex patterns for Stage 1 subtitle noise removal
HEADER_PATTERN = re.compile(r"^--- MOVIE_ID: (opus_\d+) ---$")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
BRACKET_NOISE_PATTERN = re.compile(r"\[[^\]]*\]|\([^\)]*\)")
MUSIC_SYMBOL_PATTERN = re.compile(r"[♪♫#]+")
SPEAKER_DASH_PATTERN = re.compile(r"(?:^|\s)-\s+")
WATERMARK_PATTERN = re.compile(
    r"(?:subtitles?\s+by|synced\s+and\s+corrected\s+by|encoded\s+by|www\.\S+|https?://\S+)",
    re.IGNORECASE,
)
# Repairs tokenized spacing created when OPUS XML <w> tags were joined with spaces
DETOKENIZE_PUNCT_PATTERN = re.compile(r"\s+([.,!?;:%\)\]\}])")
DETOKENIZE_OPEN_PATTERN = re.compile(r"([\(\[\{])\s+")
DETOKENIZE_APOSTROPHE_PATTERN = re.compile(r"\s+('(?:s|t|d|re|ve|ll|m)|n't)\b", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_spacy_model() -> spacy.Language:
    """Loads en_core_web_sm with parser/NER disabled, auto-downloading if missing."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        print("spaCy model 'en_core_web_sm' not found. Downloading automatically...")
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    nlp.max_length = 2_000_000
    return nlp


def clean_subtitle_noise(raw_text: str) -> str:
    """Stage 1: Removes subtitle artifacts while preserving casing and sentence structure."""
    if not raw_text or not isinstance(raw_text, str):
        return ""

    text = unicodedata.normalize("NFKC", raw_text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = BRACKET_NOISE_PATTERN.sub(" ", text)
    text = MUSIC_SYMBOL_PATTERN.sub(" ", text)
    text = WATERMARK_PATTERN.sub(" ", text)
    text = SPEAKER_DASH_PATTERN.sub(" ", text)

    # Restore natural punctuation spacing from OPUS XML <w> token joins
    text = DETOKENIZE_PUNCT_PATTERN.sub(r"\1", text)
    text = DETOKENIZE_OPEN_PATTERN.sub(r"\1", text)
    text = DETOKENIZE_APOSTROPHE_PATTERN.sub(r"\1", text)

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_short_metadata(text: object) -> str:
    """Normalizes short metadata text fields (title, plot) for dense embeddings."""
    if pd.isna(text) or not str(text).strip():
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    return WHITESPACE_PATTERN.sub(" ", cleaned).strip()


def build_combined_embedding_text(row: pd.Series) -> str:
    """Combines structured metadata and cleaned dialogue for dense transformer encoders."""
    parts: list[str] = []

    title = str(row.get("tmdb_title", "")).strip()
    year = str(row.get("release_year", "")).strip()
    if title:
        header = f"Title: {title} ({year})" if year and year != "nan" else f"Title: {title}"
        parts.append(header)

    genres = str(row.get("genres", "")).replace("|", ", ").strip()
    if genres:
        parts.append(f"Genres: {genres}")

    keywords = str(row.get("keywords", "")).replace("|", ", ").strip()
    if keywords:
        parts.append(f"Keywords: {keywords}")

    plot = str(row.get("clean_plot", "")).strip()
    if plot:
        parts.append(f"Plot: {plot}")

    subtitles = str(row.get("clean_subtitles", "")).strip()
    if subtitles:
        parts.append(f"Dialogue: {subtitles}")

    return " | ".join(parts)


def stream_target_subtitles(txt_path: str, target_ids: set[str]) -> Iterator[tuple[str, str]]:
    """Streams raw_subtitles.txt line-by-line and yields (opus_id, raw_text) for matched movies only."""
    current_id: str | None = None
    is_target = False
    buffer: list[str] = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            match = HEADER_PATTERN.match(stripped)

            if match:
                if current_id is not None and is_target:
                    yield current_id, " ".join(buffer)

                current_id = match.group(1)
                is_target = current_id in target_ids
                buffer = []
            elif is_target and stripped:
                buffer.append(stripped)

        if current_id is not None and is_target:
            yield current_id, " ".join(buffer)


def lemmatize_for_tfidf(
    texts: list[str],
    nlp: spacy.Language,
    batch_size: int = 256,
) -> list[str]:
    """Track A: Single-process batched lemmatization (Windows-safe) with live tqdm ETA."""
    processed_docs: list[str] = []
    pipe_stream = nlp.pipe(texts, batch_size=batch_size, n_process=1)

    for doc in tqdm(
        pipe_stream,
        total=len(texts),
        desc="Lemmatizing for TF-IDF",
        unit="movie",
        smoothing=0.05,
    ):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.is_alpha
            and len(token.lemma_) > 1
        ]
        processed_docs.append(" ".join(lemmas))

    return processed_docs


def run_edge_case_tests(nlp: spacy.Language) -> None:
    """Verifies cleaning logic against malformed subtitles, HTML tags, watermarks, and empty inputs."""
    cases = {
        "html_and_hi_tags": "<i>- How 'd you score that ?</i> [GUNSHOTS] (Sighs) ♪ La la la ♪",
        "watermark": "Subtitles by OpenSubtitles.org www.opensubtitles.org Synced and corrected by User123",
        "empty_input": "",
        "numbers_and_punct": "1999 !!! ??? ... 42",
    }

    print("Running preprocessing edge-case tests...")
    cleaned_stage1 = {k: clean_subtitle_noise(v) for k, v in cases.items()}
    # Use 1 worker for the 4-item micro test to avoid multiprocessing startup overhead
    tfidf_out = lemmatize_for_tfidf(list(cleaned_stage1.values()), nlp, batch_size=4)

    for (name, stage1_val), tfidf_val in zip(cleaned_stage1.items(), tfidf_out):
        print(f"  [{name}]")
        print(f"    Embedding track: '{stage1_val}'")
        print(f"    TF-IDF track:    '{tfidf_val}'")
    print("Edge-case verification complete.\n")


def run_preprocessing(min_subtitle_words: int = 50) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_txt_path = os.path.join(script_dir, "..", "data", "raw", "raw_subtitles.txt")
    enriched_path = os.path.join(script_dir, "..", "data", "processed", "enriched_metadata.csv")
    parquet_out_path = os.path.join(script_dir, "..", "data", "processed", "clean_movies.parquet")
    sample_csv_path = os.path.join(script_dir, "..", "data", "processed", "sample_clean_movies.csv")

    if not os.path.exists(enriched_path):
        raise FileNotFoundError(f"Missing {enriched_path}. Run enrich_tmdb_details.py first.")
    if not os.path.exists(raw_txt_path):
        raise FileNotFoundError(f"Missing {raw_txt_path}. Run extract_opus_metadata.py first.")

    print("Loading spaCy English model...")
    nlp = load_spacy_model()

    run_edge_case_tests(nlp)

    print("Loading enriched metadata...")
    meta_df = pd.read_csv(enriched_path, sep=";", encoding="utf-8-sig")
    meta_df = meta_df.drop_duplicates(subset=["imdb_id"], keep="first").copy()

    # Standardize missing metadata fields
    for col in ["tmdb_title", "genres", "keywords", "plot", "poster_url"]:
        if col in meta_df.columns:
            meta_df[col] = meta_df[col].fillna("")

    meta_df["release_year"] = (
        pd.to_numeric(meta_df["release_year"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .replace("0", "")
    )
    meta_df["clean_plot"] = meta_df["plot"].apply(clean_short_metadata)

    target_opus_ids = set(meta_df["opus_id"].dropna().astype(str).tolist())
    print(f"Streaming {raw_txt_path} for {len(target_opus_ids):,} aligned movies...")

    subtitle_map: dict[str, str] = {}
    stream_iter = stream_target_subtitles(raw_txt_path, target_opus_ids)
    for opus_id, raw_sub in tqdm(
        stream_iter,
        total=len(target_opus_ids),
        desc="Stage 1 cleaning",
        unit="movie",
    ):
        subtitle_map[opus_id] = clean_subtitle_noise(raw_sub)

    meta_df["clean_subtitles"] = meta_df["opus_id"].map(subtitle_map).fillna("")
    meta_df["plot_word_count"] = meta_df["clean_plot"].str.split().str.len()
    meta_df["subtitle_word_count"] = meta_df["clean_subtitles"].str.split().str.len()
    meta_df["has_valid_subtitles"] = meta_df["subtitle_word_count"] >= min_subtitle_words

    # Track B: Dense Semantic Embedding input (preserves natural casing and sentence structure)
    print("Building dense embedding text representations...")
    meta_df["text_embedding"] = meta_df.apply(build_combined_embedding_text, axis=1)

    # Track A: Lexical TF-IDF input (multi-core lemmatized, stopword-filtered, lowercase)
    print("Lemmatizing plot + subtitle corpus for TF-IDF...")
    raw_lexical_corpus = (
        meta_df["tmdb_title"]
        + " "
        + meta_df["genres"].str.replace("|", " ", regex=False)
        + " "
        + meta_df["keywords"].str.replace("|", " ", regex=False)
        + " "
        + meta_df["clean_plot"]
        + " "
        + meta_df["clean_subtitles"]
    ).tolist()
    meta_df["text_tfidf"] = lemmatize_for_tfidf(raw_lexical_corpus, nlp, batch_size=128)

    # Save full master dataset as Parquet (preserves multi-thousand-word strings without delimiter issues)
    os.makedirs(os.path.dirname(parquet_out_path), exist_ok=True)
    meta_df.to_parquet(parquet_out_path, index=False)
    print(f"Saved full preprocessed dataset ({len(meta_df):,} rows) to {parquet_out_path}")

    # Export a 100-row sample CSV for GitHub handoff and Excel inspection
    valid_sample = meta_df[meta_df["has_valid_subtitles"]].head(100)
    valid_sample.to_csv(sample_csv_path, index=False, sep=";", encoding="utf-8-sig")
    print(f"Saved 100-row team handoff sample to {sample_csv_path}")


if __name__ == "__main__":
    run_preprocessing(min_subtitle_words=50)