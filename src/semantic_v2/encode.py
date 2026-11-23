"""Sentence encoders: model registry lookup, prefixes, fp16 on GPU, offline debug stand-in."""
from __future__ import annotations

import logging

import numpy as np

from .config import MODEL_REGISTRY, SemanticV2Error

LOGGER = logging.getLogger("semantic_v2")


def _l2(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(n == 0, 1, n)


class _HashingEncoder:
    """Hashing bag-of-words. ONLY for offline tests of the pipeline: not semantic."""

    def __init__(self, dim: int = 256):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.dim = dim
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 1))

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


class TextEncoder:
    def __init__(self, key: str, *, device: str | None = None, fp16: bool = True, batch_size: int = 64):
        if key not in MODEL_REGISTRY:
            raise SemanticV2Error(f"Unknown model {key!r}; choose from {sorted(MODEL_REGISTRY)}.")
        self.key, self.spec, self.batch_size = key, MODEL_REGISTRY[key], batch_size
        if key == "debug-hashing":
            LOGGER.warning("Using the debug hashing encoder: results are NOT semantic.")
            self._model = _HashingEncoder()
            self.dim = self._model.dim
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticV2Error("Install: pip install sentence-transformers torch (CUDA build for the GPU).") from exc
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading %s on %s", self.spec.hf_name, device)
        self._model = SentenceTransformer(self.spec.hf_name, device=device)
        self._model.max_seq_length = min(self.spec.max_tokens, self._model.max_seq_length or self.spec.max_tokens)
        if device == "cuda" and fp16:
            self._model.half()
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if isinstance(self._model, _HashingEncoder):
            return _l2(self._model.encode(texts))
        emb = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
        return _l2(emb)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.spec.doc_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts])


def load_encoders(field_key: str, chunk_key: str, *, batch_size: int = 64, device: str | None = None):
    """Return (field_encoder, chunk_encoder); the same object when both keys are equal."""
    fe = TextEncoder(field_key, batch_size=batch_size, device=device)
    ce = fe if chunk_key == field_key else TextEncoder(chunk_key, batch_size=batch_size, device=device)
    return fe, ce
