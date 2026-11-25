"""Configuration for semantic search v2.

``BuildConfig``  : everything that changes the *index* (a change means re-encoding).
``SearchConfig`` : runtime parameters only (a change never needs re-encoding).
``QueryConfig``  : query processing (normalisation, spelling, negation, clauses).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


class SemanticV2Error(Exception):
    """Actionable input, build or index error."""


@dataclass(frozen=True)
class ModelSpec:
    hf_name: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_tokens: int = 256
    note: str = ""


# key -> spec. `max_tokens` is the window the model was trained for; verify with model.max_seq_length.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "minilm-l6": ModelSpec("sentence-transformers/all-MiniLM-L6-v2", max_tokens=256, note="v1 baseline encoder"),
    "minilm-l12": ModelSpec("sentence-transformers/all-MiniLM-L12-v2", max_tokens=128,
                            note="128-token window: 150-word chunks would be truncated"),
    "bge-small": ModelSpec("BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: ",
                           max_tokens=512),
    "bge-base": ModelSpec("BAAI/bge-base-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: ",
                          max_tokens=512),
    "e5-small": ModelSpec("intfloat/e5-small-v2", query_prefix="query: ", doc_prefix="passage: ", max_tokens=512),
    "e5-base": ModelSpec("intfloat/e5-base-v2", query_prefix="query: ", doc_prefix="passage: ", max_tokens=512),
    "gte-small": ModelSpec("thenlper/gte-small", max_tokens=512),
    "gte-base": ModelSpec("thenlper/gte-base", max_tokens=512),
    "mpnet": ModelSpec("sentence-transformers/all-mpnet-base-v2", max_tokens=384),
    "debug-hashing": ModelSpec("debug-hashing", max_tokens=100000, note="offline stand-in, never report its results"),
}

FIELDS = ("plot", "keywords", "genres")
SOURCES = (*FIELDS, "subtitles")


@dataclass
class BuildConfig:
    field_model: str = "minilm-l6"       # encodes plot / keywords / genres / hub bank (short texts)
    chunk_model: str = "minilm-l6"       # encodes subtitle chunks (the expensive part)
    chunk_words: int = 150
    chunk_overlap: int = 0
    max_chunks_per_movie: int = 0        # 0 = all; >0 = evenly spaced subset (documented fallback)
    min_dialogue_words: int = 50         # shorter dialogue is treated as "no subtitles"
    max_plot_words: int = 250
    max_keywords: int = 15
    keywords_template: str = "Keywords: {}"
    genres_template: str = "Genres: {}"
    shard_movies: int = 2000             # unit of resumable work
    batch_size: int = 64
    hub_bank_templated: int = 400        # background queries for hub correction (never evaluation queries)
    hub_bank_plot_sentences: int = 400
    hub_k: int = 10                      # mean of the k highest bank scores = movie "hubness"
    agg_topk: int = 3                    # subtitle score = mean of the k best chunks per movie
    seed: int = 13

    def validate(self) -> None:
        for key in (self.field_model, self.chunk_model):
            if key not in MODEL_REGISTRY:
                raise SemanticV2Error(f"Unknown model {key!r}; choose from {sorted(MODEL_REGISTRY)}.")
        if self.chunk_words < 10 or not 0 <= self.chunk_overlap < self.chunk_words:
            raise SemanticV2Error("chunk_words >= 10 and 0 <= chunk_overlap < chunk_words required.")
        limit = MODEL_REGISTRY[self.chunk_model].max_tokens
        if self.chunk_words * 1.4 > limit:  # ~1.3-1.4 word pieces per English word
            raise SemanticV2Error(
                f"chunk_words={self.chunk_words} (~{int(self.chunk_words * 1.4)} tokens) exceeds the "
                f"{limit}-token window of {self.chunk_model}. Use smaller chunks or another model.")
        if self.shard_movies < 1 or self.agg_topk < 1 or self.hub_k < 1:
            raise SemanticV2Error("shard_movies, agg_topk and hub_k must be >= 1.")

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class SearchConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "plot": 1.0, "keywords": 0.4, "genres": 0.2, "subtitles": 0.5})   # placeholders until tuned on dev
    hub_fields: tuple[str, ...] = ("subtitles",)   # sources that get the hub correction
    hub_gamma: float = 1.0                         # s' = s - gamma * hubness(movie)
    alpha: float = 0.5                             # weight of the whole query vs. the soft-min over clauses
    softmin_T: float = 1.0
    lam: float = 1.0                               # negation penalty strength
    tau: float = 0.5                               # z-score a negated term may reach before penalising
    neg_template: str = "{term}"
    meta_negation: bool = False                    # D8: genre/keyword based negation (ablation)
    meta_neg_penalty: float = 2.0
    use_clauses: bool = True
    min_confidence: float = 0.0                    # abstain below this raw weighted cosine (0 = off; calibrate on dev)

    # ---- JSON round trip: the webapp loads one settings file instead of a dozen flags
    def to_dict(self) -> dict:
        d = asdict(self)
        d["hub_fields"] = list(self.hub_fields)
        return d

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "SearchConfig":
        d = {k: v for k, v in d.items() if not k.startswith("_")}          # "_comment" keys are allowed
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise SemanticV2Error(f"Unknown SearchConfig keys {sorted(unknown)}.")
        if "hub_fields" in d:
            d["hub_fields"] = tuple(d["hub_fields"])
        if "weights" in d:
            d["weights"] = {**cls().weights, **d["weights"]}
        cfg = cls(**d)
        cfg.validate()
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
        except json.JSONDecodeError as exc:
            raise SemanticV2Error(f"{path} is not valid JSON: {exc}") from exc

    def validate(self) -> None:
        bad = set(self.weights) - set(SOURCES)
        if bad:
            raise SemanticV2Error(f"Unknown weight keys {sorted(bad)}.")
        if not any(w > 0 for w in self.weights.values()):
            raise SemanticV2Error("At least one source weight must be positive.")
        if not 0 <= self.alpha <= 1 or self.softmin_T <= 0 or self.lam < 0:
            raise SemanticV2Error("alpha in [0,1], softmin_T > 0, lam >= 0 required.")


@dataclass
class QueryConfig:
    max_words: int = 60              # warn above this
    hard_max_words: int = 150        # truncate above this
    spell: bool = True
    spell_max_unknown_fraction: float = 0.6   # if more tokens are unknown, assume gibberish: do not "correct" it
    min_corpus_count: int = 3        # corpus words rarer than this are not trusted as "known"
    clauses: bool = True
    max_clauses: int = 4
    max_clause_words: int = 5
