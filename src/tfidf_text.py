"""TF-IDF query processing aligned to Vojta's existing text_tfidf corpus.

The existing movie texts were created by preprocessing.py's
lemmatize_for_tfidf: spaCy English tokenization, alphabetic-only tokens,
spaCy English STOP_WORDS, lowercase, and NLTK WordNet noun lemmatization.

This module uses the same lexical steps and removes a short set of
*query-initial* conversational expressions before applying them.

IMPORTANT: existing text_tfidf removes negation words like 'not'. This
baseline cannot reliably express exclusions (e.g. 'dark but not scary').
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


# Only remove preambles at the START of the query. Never delete important
# words from the middle of movie-related descriptions.
START_PHRASES = [
    r"(?:hi|hello|hey)[,!.\s]+",
    r"(?:can|could|would)\s+you\s+(?:please\s+)?(?:recommend|suggest)(?:\s+me)?\s*",
    r"(?:please\s+)?(?:recommend|suggest)(?:\s+me)?\s*",
    r"i\s+(?:feel\s+like|want\s+to|would\s+like\s+to|wanna|would\s+love\s+to)\s+(?:watch(?:ing)?|see(?:ing)?)\s*",
    r"i(?:'m|\s+am)\s+in\s+the\s+mood\s+for\s*",
    r"i(?:'d|\s+would)\s+like\s+(?:to\s+watch\s+)?",
    r"i\s+(?:want|need)\s+(?:to\s+(?:watch|see)\s+)?",
]
LEADING_GENERIC = [
    r"(?:something|anything|a|an|some|the)\s+",
    r"(?:movie|movies|film|films)\s+",
    r"(?:that\s+(?:is|feels|has)\s+)",
]


def strip_query_filler(query: str) -> str:
    """Remove polite conversational framing, not content-bearing mood terms."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string")
    text = unicodedata.normalize("NFKC", query).replace("’", "'").strip().lower()
    # Allow several initial expressions, e.g. 'hey, can you recommend me something ...'
    for _ in range(6):
        previous = text
        for pattern in START_PHRASES:
            text = re.sub(r"^\s*" + pattern, "", text, count=1, flags=re.IGNORECASE)
        for pattern in LEADING_GENERIC:
            text = re.sub(r"^\s*" + pattern, "", text, count=1, flags=re.IGNORECASE)
        text = text.lstrip(" ,.!?;:")
        if text == previous:
            break
    return text.strip()


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from spacy.lang.en import English
    except ImportError as exc:
        raise RuntimeError("spaCy is required for consistent TF-IDF preprocessing: pip install spacy") from exc
    # English() uses the same default language tokenizer rules as en_core_web_sm,
    # without loading slow or unnecessary model components.
    return English().tokenizer


@lru_cache(maxsize=1)
def _lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer
        lemma = WordNetLemmatizer()
        lemma.lemmatize("movies")  # fail early if the WordNet resource is absent
    except (ImportError, LookupError) as exc:
        raise RuntimeError(
            "WordNet is missing. Install nltk and run: "
            "python -m nltk.downloader wordnet omw-1.4"
        ) from exc
    return lemma


def preprocess_tfidf_text(text: str) -> str:
    """
    Do not use this to reprocess already-normalized text_tfidf columns.
    In a future dataset rebuild, the team can call this shared function
    for both full movie documents and user queries.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    from spacy.lang.en.stop_words import STOP_WORDS
    tokenizer = _tokenizer()
    lemma = _lemmatizer()
    words = []
    for token in tokenizer(unicodedata.normalize("NFKC", text)):
        if token.is_alpha and not token.is_space:
            word = token.text.lower()
            if word not in STOP_WORDS and len(word) > 1:
                base = lemma.lemmatize(word)
                if len(base) > 1 and base not in STOP_WORDS:
                    words.append(base)
    return " ".join(words)


def preprocess_tfidf_query(query: str) -> str:
    """Strip conversational filler, then match document normalization exactly."""
    return preprocess_tfidf_text(strip_query_filler(query))
