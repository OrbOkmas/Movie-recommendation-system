"""Query processing: normalisation, length warning, spell correction, negation parsing, clause splitting.

``QueryProcessor.process`` never raises on user input. It returns a ``ProcessedQuery`` whose ``status`` is one of
``ok`` | ``ok_with_warning`` | ``empty`` | ``negation_only`` | ``low_confidence`` (the last one is also set later by
the scorer when the best match is too weak).

Design principle: every step is conservative. If the parser is unsure it leaves the query alone, because the
whole query is always scored as well (see scoring.py).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import QueryConfig
from .lexicon import (AUX, COPULA, FILLER_PATTERNS, GENERIC_NOUNS, LEADING_JUNK, LIKE_VERBS, NEG_CUES, PROTECTED_WORDS,
                      SPAN_BOUNDARY, SPAN_LIST_SEP, SPAN_STOP, TRAILING_JUNK, VAGUE_WORDS)

LOGGER = logging.getLogger("semantic_v2")
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?(?:-[a-z0-9]+)*|[,.;:!?]")
PUNCT = {",", ".", ";", ":", "!", "?"}
COPULA_CONTRACTIONS = {"isn't", "aren't", "wasn't", "weren't", "ain't"}
FINITE_TAGS = {"VB", "VBD", "VBP", "VBZ", "MD"}
SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+(?:and|but(?:\s+also)?|yet|with|while|plus|though|as\s+well\s+as)\s+")
EXTRA_LIKE = LIKE_VERBS | {"stand", "bear", "handle"}


@dataclass
class ProcessedQuery:
    original: str
    text: str = ""                    # normalised + spell-corrected query
    status: str = "ok"
    warnings: list[dict] = field(default_factory=list)
    whole_text: str = ""              # what is encoded as "the whole query" (negated parts removed)
    clauses: list[str] = field(default_factory=list)
    neg_terms: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def searchable(self) -> bool:
        return self.status in {"ok", "ok_with_warning"}


# ----------------------------------------------------------------------------- negation
def _is_cue(tok: str) -> bool:
    return tok in NEG_CUES or tok.endswith("n't")


def _detokenize(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        out += t if (t in PUNCT or not out) else " " + t
    return out


def _strip_edges(tokens: list[str]) -> list[str]:
    toks = list(tokens)
    changed = True
    while toks and changed:
        changed = False
        if toks[-1] in PUNCT or toks[-1] in TRAILING_JUNK:
            toks.pop(); changed = True
        elif toks[0] in PUNCT or toks[0] in LEADING_JUNK:
            toks.pop(0); changed = True
    return toks


def parse_negation(text: str) -> tuple[str, list[str]]:
    """Return (text without the negated parts, negated terms). Unsure cases are left untouched."""
    toks = TOKEN.findall(text.lower().replace("\u2019", "'"))
    n = len(toks)
    remove = [False] * n
    terms: list[str] = []
    i = 0
    while i < n:
        t = toks[i]
        prev = toks[i - 1] if i else ""
        nxt = toks[i + 1] if i + 1 < n else ""
        start: int | None = None
        # hyphenated "-free": "gore-free"
        if t.endswith("-free") and len(t) > 6:
            terms.append(t[:-5]); remove[i] = True; i += 1; continue
        if t in {"free", "instead", "rather"} and nxt in {"of", "than"} and (t != "free" or prev not in COPULA):
            start = i + 2
        elif t == "not":
            if nxt in {"only", "just"}:
                start = None
            elif prev in AUX and prev not in COPULA:
                start = _after_like_verb(toks, i + 1)
            else:
                start = i + 1
        elif t in COPULA_CONTRACTIONS:
            start = i + 1
        elif t.endswith("n't") or t == "cannot":
            start = _after_like_verb(toks, i + 1)
        elif t in {"no", "without", "never", "avoid", "avoiding"}:
            start = i + 1
        elif t == "nothing":
            start = None if nxt == "but" else i + 1
        if start is None:
            i += 1
            continue
        j, cur, found = start, [], []
        while j < n:
            tj = toks[j]
            if tj in SPAN_BOUNDARY or _is_cue(tj):
                break
            if tj in SPAN_LIST_SEP:
                if cur:
                    found.append(" ".join(cur)); cur = []
                j += 1
                continue
            if tj in SPAN_STOP:
                j += 1
                continue
            cur.append(tj)
            j += 1
            if len(cur) >= 4:
                break
        if cur:
            found.append(" ".join(cur))
        if not found:            # a cue with nothing after it ("no.", "not at all"): leave it alone
            i += 1
            continue
        terms.extend(found)
        for k in range(i, j):
            remove[k] = True
        i = j
    if not terms:
        return text, []
    kept = _strip_edges([t for t, r in zip(toks, remove) if not r])
    text_out = _detokenize(kept)
    text_out = re.sub(r"\b(?:but|and|yet|or)\s*(?=[,.;:!?])", "", text_out)       # "funny but, with x" -> "funny, with x"
    text_out = re.sub(r"\s*,(\s*,)+", ",", text_out).replace(" ,", ",").strip(" ,")
    return text_out, terms


def _after_like_verb(toks: list[str], k: int) -> int | None:
    """After "do not"/"don't": only treat as negation if a wanting/liking verb follows ("don't want violence")."""
    if k < len(toks) and toks[k] in EXTRA_LIKE:
        k += 1
        if k < len(toks) and toks[k] == "to":
            k += 1
            if k < len(toks) and toks[k] in {"see", "watch", "have", "feel", "be"}:
                k += 1
        return k
    return None


# ----------------------------------------------------------------------------- clauses
def strip_filler(text: str) -> str:
    t = " ".join(text.strip().lower().split())
    t = re.sub(r"[.!?\s]+$", "", t)
    prev = None
    while prev != t:
        prev = t
        for pat in FILLER_PATTERNS:
            t = re.sub(pat, "", t).strip()
    return t


def split_clauses(text: str, nlp, cfg: QueryConfig) -> list[str]:
    """Split enumerations of short attribute phrases ("cozy and a little grim"); never narrative sentences."""
    if nlp is None:
        return []
    t = strip_filler(text)
    if not t:
        return []
    doc = nlp(t)
    if any(tok.tag_ in FINITE_TAGS for tok in doc):
        return []                                    # finite verb -> narrative query -> do not split
    if any(tok.tag_ in {"IN", "TO"} and tok.lower_ != "with" for tok in doc):
        return []                                    # prepositions -> topical phrase ("about grief and loss")
    pieces = []
    for p in SPLIT_RE.split(t):
        p = re.sub(r"^(?:with|and|but|also)\s+", "", re.sub(r"[.!?,;]+$", "", p.strip()).strip())
        words = p.split()
        if len(words) > 1 and words[-1] in GENERIC_NOUNS:
            p = " ".join(words[:-1])
        if p:
            pieces.append(p)
    if not 2 <= len(pieces) <= cfg.max_clauses or any(len(p.split()) > cfg.max_clause_words for p in pieces):
        return []
    if any(re.fullmatch(r"(?:a|an|the)\s+\w+", p) for p in pieces):
        return []                                    # dangling "a beautiful" -> the split cut an adjective pair
    return pieces


# ----------------------------------------------------------------------------- spelling
class SpellCorrector:
    def __init__(self, vocab: dict[str, int] | None = None, min_corpus_count: int = 3):
        try:
            import symspellpy
            from symspellpy import SymSpell
        except ImportError as exc:
            raise RuntimeError("pip install symspellpy") from exc
        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        path = Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt"
        self.sym.load_dictionary(str(path), term_index=0, count_index=1)
        self.domain = {w for w, c in (vocab or {}).items() if c >= min_corpus_count and w.isalpha()}
        for w in self.domain:
            if len(w) >= 3:
                self.sym.create_dictionary_entry(w, max(int(vocab[w]), 1))   # type: ignore[index]
        for w in PROTECTED_WORDS:
            self.sym.create_dictionary_entry(w, 50_000)

    def known(self, w: str) -> bool:
        return w in self.sym.words or w in self.domain or w in PROTECTED_WORDS

    @staticmethod
    def _variant_key(w: str) -> str:
        """US/UK spelling variants (humor/humour, realize/realise, theater/theatre) share one key."""
        w = re.sub(r"our\b", "or", w)
        w = re.sub(r"is(e|ed|es|ing|ation)\b", r"iz\1", w)
        return re.sub(r"re\b", "er", w)

    def _candidate(self, w: str) -> str | None:
        if re.search(r"(.)\1\1", w):                         # elongation: gloooomy
            for rep in (r"\1\1", r"\1"):
                c = re.sub(r"(.)\1{2,}", rep, w)
                if self.known(c):
                    return c
        if len(w) <= 3:
            return None
        dist = 1 if len(w) <= 5 else 2
        from symspellpy import Verbosity
        hits = [h for h in self.sym.lookup(w, Verbosity.ALL, max_edit_distance=dist) if h.term != w]
        same_first = [h for h in hits if h.term[0] == w[0]]      # typos rarely change the first letter
        hits = same_first or []
        if not hits:
            return None
        def pure_substitution(h) -> bool:
            return len(h.term) == len(w) and sorted(h.term) != sorted(w)

        # distance 2 is only trusted for missing/extra letters or transpositions, not for two substitutions
        hits = [h for h in hits if h.distance < 2 or not pure_substitution(h)]
        if not hits:
            return None
        best = min(hits, key=lambda h: (h.distance, pure_substitution(h), -h.count))
        if self._variant_key(best.term) == self._variant_key(w):
            return None
        return best.term

    def correct(self, text: str, max_unknown_fraction: float) -> tuple[str, dict[str, str], bool]:
        """Returns (text, corrections, looks_like_gibberish)."""
        words = [m for m in re.finditer(r"[A-Za-z][A-Za-z']*", text)]
        checkable = [m for m in words if "'" not in m.group() and len(m.group()) >= 3]
        unknown = [m for m in checkable if not self.known(m.group().lower())]
        if not checkable or not unknown:
            return text, {}, False
        cands = {m: self._candidate(m.group().lower()) for m in unknown}
        uncorrectable = sum(1 for c in cands.values() if not c)
        if uncorrectable / len(checkable) > max_unknown_fraction:   # mostly gibberish: do not invent words
            return text, {}, True
        fixes: dict[str, str] = {}
        out, last = [], 0
        for m, cand in cands.items():
            if not cand:
                continue
            w = m.group()
            cand = cand.upper() if w.isupper() and len(w) > 1 else cand.capitalize() if w[0].isupper() else cand
            fixes[w] = cand
            out.append(text[last:m.start()]); out.append(cand); last = m.end()
        out.append(text[last:])
        return "".join(out), fixes, False


# ----------------------------------------------------------------------------- processor
class QueryProcessor:
    def __init__(self, cfg: QueryConfig | None = None, vocab: dict[str, int] | None = None, *, use_spacy: bool = True):
        self.cfg = cfg or QueryConfig()
        self.spell = None
        if self.cfg.spell:
            try:
                self.spell = SpellCorrector(vocab, self.cfg.min_corpus_count)
            except RuntimeError as exc:
                LOGGER.warning("Spell correction disabled: %s", exc)
        self.nlp = None
        if use_spacy and self.cfg.clauses:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Clause splitting disabled (spaCy model en_core_web_sm unavailable): %s", exc)

    def process(self, query) -> ProcessedQuery:
        pq = ProcessedQuery(original=query if isinstance(query, str) else repr(query))
        gibberish = False
        if not isinstance(query, str):
            pq.status = "empty"; pq.warn("not_a_string", "The query must be text."); return pq
        text = unicodedata.normalize("NFKC", query).replace("\u2019", "'").replace("\u2018", "'")
        text = " ".join(text.split())
        if not text:
            pq.status = "empty"; pq.warn("empty_query", "The query is empty."); return pq
        letters = [c for c in text if c.isalpha()]
        if not letters:
            pq.text = pq.whole_text = text
            pq.status = "low_confidence"; pq.warn("no_text", "The query has no words to search for."); return pq
        if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            text = text.lower()                                   # SHOUTING -> normal case (silent)
        text = re.sub(r"([!?.,;:])\1+", r"\1", text)
        words = text.split()
        if len(words) > self.cfg.max_words:
            pq.warn("query_too_long", f"The query has {len(words)} words (limit {self.cfg.max_words}); long "
                    "queries are diluted and the encoder only reads the first part. Shorter is better.")
        if len(words) > self.cfg.hard_max_words:
            text = " ".join(words[:self.cfg.hard_max_words])
            pq.warn("query_truncated", f"Only the first {self.cfg.hard_max_words} words were used.")
        if self.spell is not None:
            text, fixes, gibberish = self.spell.correct(text, self.cfg.spell_max_unknown_fraction)
            if fixes:
                pq.corrections = fixes
                pq.warn("spelling_corrected", "Corrected: " + ", ".join(f"{a} -> {b}" for a, b in fixes.items()))
        pq.text = text
        pos_text, terms = parse_negation(text)
        pq.neg_terms = terms
        pq.whole_text = pos_text if terms else text
        if terms and not strip_filler(pos_text).strip(" ,.;:!?"):
            pq.status = "negation_only"
            pq.warn("negation_only", "The query only says what you do NOT want; describe what you do want.")
            return pq
        if gibberish:
            pq.status = "low_confidence"
            pq.warn("unrecognized_words", "Most words in the query were not recognised; check the spelling.")
            return pq
        content = [w for w in re.findall(r"[a-z][a-z'-]*", pos_text.lower()) if w not in VAGUE_WORDS]
        if not content:
            pq.status = "low_confidence"
            pq.warn("too_vague", "The query does not say what you are looking for; describe a mood, theme or story.")
            return pq
        pq.clauses = split_clauses(pos_text if terms else text, self.nlp, self.cfg) if self.cfg.clauses else []
        pq.status = "ok_with_warning" if pq.warnings else "ok"
        return pq
