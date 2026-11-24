"""Public search API: ``SemanticSearcher.search(query) -> SearchResponse`` (never raises on user input)."""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import QueryConfig, SearchConfig, SemanticV2Error
from .encode import TextEncoder, load_encoders
from .index import MovieIndex
from .query import ProcessedQuery, QueryProcessor
from .scoring import MetaNegation, Prepared, fuse, raw_scores

LOGGER = logging.getLogger("semantic_v2")


@dataclass
class SearchResponse:
    status: str
    query: dict
    warnings: list[dict] = field(default_factory=list)
    confidence: float | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message, "confidence": self.confidence,
                "warnings": self.warnings, "query": self.query, "results": self.results}


MESSAGES = {
    "empty": "Please type what kind of movie you are in the mood for.",
    "negation_only": "I only know what you do NOT want. Tell me what you do want (a mood, theme or plot).",
    "low_confidence": "No confident match: the request is too vague or I could not understand it. "
                      "Try describing a mood, theme or story.",
}


class SemanticSearcher:
    def __init__(self, index: MovieIndex, search_cfg: SearchConfig | None = None, query_cfg: QueryConfig | None = None,
                 *, encoders: tuple[TextEncoder, TextEncoder] | None = None, processor: QueryProcessor | None = None,
                 device: str | None = None):
        self.index = index
        self.cfg = search_cfg or SearchConfig()
        self.cfg.validate()
        self.fe, self.ce = encoders or load_encoders(index.cfg.field_model, index.cfg.chunk_model, device=device)
        self.processor = processor or QueryProcessor(query_cfg or QueryConfig(), index.vocab)
        self._meta = MetaNegation(index.movies)
        self._starts = [sh.m0 for sh in index.shards]

    @classmethod
    def from_dir(cls, index_dir: str | Path, search_cfg: SearchConfig | None = None, query_cfg: QueryConfig | None = None,
                 **kw) -> "SemanticSearcher":
        return cls(MovieIndex.load(index_dir), search_cfg, query_cfg, **kw)

    # ---- stage A -------------------------------------------------------------------------------------
    def prepare(self, pq: ProcessedQuery) -> Prepared:
        texts = [pq.whole_text]
        clauses = pq.clauses if self.cfg.use_clauses else []
        texts += clauses
        texts += [self.cfg.neg_template.format(term=t) for t in pq.neg_terms]
        fq = self.fe.encode_queries(texts)
        cq = fq if self.ce is self.fe else self.ce.encode_queries(texts)
        return Prepared(texts=texts, n_clauses=len(clauses), n_neg=len(pq.neg_terms),
                        raw=raw_scores(self.index, fq, cq), whole_chunk_vec=cq[0])

    # ---- stage B + formatting ------------------------------------------------------------------------
    def rank(self, pq: ProcessedQuery, prep: Prepared, *, top_k: int = 5, exclude_imdb_id: str | None = None,
             cfg: SearchConfig | None = None, snippets: bool = True) -> tuple[list[dict], float]:
        cfg = cfg or self.cfg
        out = fuse(prep, self.index, cfg, neg_terms=pq.neg_terms, meta=self._meta)
        score = out["score"]
        valid = ~np.isnan(score)
        if exclude_imdb_id is not None:
            valid &= self.index.movies["imdb_id"].to_numpy() != str(exclude_imdb_id)
        idx = np.flatnonzero(valid)
        idx = idx[np.lexsort((idx, -score[idx]))][:top_k]
        results = []
        mv = self.index.movies
        for rank, i in enumerate(idx, 1):
            row = mv.iloc[int(i)]
            rec: dict[str, Any] = {"rank": rank, "movie_id": str(row["imdb_id"]), "title": str(row["title"]),
                                   "year": str(row["year"]), "genres": str(row["genres"]),
                                   "score": round(float(score[i]), 5), "z_whole": round(float(out["z_whole"][i]), 5),
                                   "neg_penalty": round(float(out["penalty"][i]), 5)}
            for src, key in (("plot", "plot_score"), ("keywords", "keywords_score"), ("genres", "genres_score"),
                             ("subtitles", "subtitle_score")):
                v = prep.raw[src][0, i]
                rec[key] = None if np.isnan(v) else round(float(v), 5)
            if snippets and prep.whole_chunk_vec is not None and self.index.field_mask["subtitles"][i]:
                s = bisect.bisect_right(self._starts, int(i)) - 1
                sh = self.index.shards[s]
                rec["best_chunk"] = sh.best_chunk_text(int(i) - sh.m0, prep.whole_chunk_vec)
            results.append(rec)
        return results, float(out["confidence"])

    def search(self, query, *, top_k: int = 5, exclude_imdb_id: str | None = None, return_all: bool = False,
               snippets: bool = True) -> SearchResponse:
        """Never raises on user input. ``return_all`` also returns results when the status says "no match"."""
        try:
            if not isinstance(top_k, int) or top_k < 1:
                raise SemanticV2Error("top_k must be a positive integer.")
            pq = self.processor.process(query)
            scorable_anyway = return_all and pq.status == "low_confidence" and bool(pq.whole_text.strip())
            if not pq.searchable and not scorable_anyway:
                return SearchResponse(pq.status, pq.to_dict(), pq.warnings, None, [], MESSAGES.get(pq.status, ""))
            prep = self.prepare(pq)
            results, conf = self.rank(pq, prep, top_k=top_k, exclude_imdb_id=exclude_imdb_id, snippets=snippets)
            status, msg = pq.status, ""
            if pq.status == "low_confidence":                     # vague / unrecognised: no results unless asked for
                msg = MESSAGES["low_confidence"]
                if not return_all:
                    results = []
            elif self.cfg.min_confidence > 0 and conf < self.cfg.min_confidence:
                status, msg = "low_confidence", MESSAGES["low_confidence"]
                if not return_all:
                    results = []
            return SearchResponse(status, pq.to_dict(), pq.warnings, round(conf, 5), results, msg)
        except SemanticV2Error:
            raise
        except Exception as exc:  # noqa: BLE001  - a user query must never crash the app
            LOGGER.exception("Unexpected search failure")
            return SearchResponse("error", {"original": str(query)}, [{"code": "internal_error", "message": str(exc)}],
                                  None, [], "Something went wrong while searching.")
