"""End-to-end tests on a tiny synthetic corpus with the offline hashing encoder (pipeline logic, not semantics)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from semantic_v2.chunks import ChunkShard
from semantic_v2.config import BuildConfig, SearchConfig, SemanticV2Error
from semantic_v2.fields import chunk_words, iter_movies, LoadStats
from semantic_v2.index import MovieIndex, build_index, hubness
from semantic_v2.search import SemanticSearcher

THEMES = {
    "Horror": ("a haunted house with ghosts and blood and gore", "ghost|gore|haunted house"),
    "Comedy": ("a funny silly farce full of jokes and slapstick", "slapstick|parody"),
    "Romance": ("two lovers fall in love in a quiet seaside town", "love|wedding"),
    "Thriller": ("a detective hunts a killer in a snowy city", "murder|detective"),
}


def make_csv(path, n=24, with_dialogue=True):
    rows = []
    for i in range(n):
        g = list(THEMES)[i % 4]
        plot, kw = THEMES[g]
        row = {"opus_id": f"opus_{i}", "imdb_id": f"tt{i:07d}", "tmdb_title": f"Movie {i} {g}", "release_year": 1980 + i,
               "genres": g, "keywords": kw if i % 5 else "", "clean_plot": plot if i % 7 else ""}
        dlg = " ".join((plot.split() * 40)[:400]) if with_dialogue else ""
        head = f"Title: x | Genres: {g}" + (f" | Plot: {plot}" if row["clean_plot"] else "")
        row["text_embedding"] = f"{head} | Dialogue: {dlg}" if dlg else head
        rows.append(row)
    rows.append({"opus_id": "o_dup", "imdb_id": "tt0000000", "tmdb_title": "Dup", "release_year": 1999, "genres": "Drama",
                 "keywords": "", "clean_plot": "dup", "text_embedding": ""})                       # duplicate id
    rows.append({"opus_id": "o_empty", "imdb_id": "tt9999999", "tmdb_title": "Nothing", "release_year": 1999,
                 "genres": "Drama", "keywords": "", "clean_plot": "", "text_embedding": ""})        # no plot/kw/dialogue
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("v2")
    make_csv(d / "data.csv")
    cfg = BuildConfig(field_model="debug-hashing", chunk_model="debug-hashing", shard_movies=10,
                      hub_bank_templated=40, hub_bank_plot_sentences=10, chunk_words=60)
    m = build_index(d / "data.csv", d / "ix", cfg)
    return d, cfg, m


@pytest.fixture(scope="module")
def searcher(built):
    d, _, _ = built
    return SemanticSearcher.from_dir(d / "ix", SearchConfig())


def test_chunking():
    assert chunk_words("a b c", 10) == ["a b c"]
    ch = chunk_words(" ".join(str(i) for i in range(320)), 150)
    assert [len(c.split()) for c in ch] == [150, 170]            # 20-word tail merged
    assert chunk_words("", 10) == []
    capped = chunk_words(" ".join(["w"] * 1500), 100, max_chunks=5)
    assert len(capped) == 5


def test_build_stats_and_exclusions(built):
    _, _, m = built
    s = m["stats"]
    assert m["complete"] and s["duplicate_ids"] == 1 and s["excluded_no_text"] == 1
    assert s["indexed_movies"] == 24 and s["n_shards"] == 3 and s["n_chunks"] > 0
    assert s["hub_bank_size"] > 40


def test_missing_fields_have_no_vectors(built):
    d, _, _ = built
    ix = MovieIndex.load(d / "ix")
    assert not ix.field_mask["plot"].all() and not ix.field_mask["keywords"].all()
    for f in ("plot", "keywords"):
        assert np.all(ix.field_emb[f][~ix.field_mask[f]] == 0)
    assert all(np.isnan(ix.hub[f][~ix.field_mask[f]]).all() for f in ("plot", "keywords", "genres", "subtitles"))
    assert not np.isnan(ix.hub["plot"][ix.field_mask["plot"]]).any()


def test_search_finds_matching_genre(searcher):
    r = searcher.search("a haunted house with ghosts", top_k=5)
    assert r.status == "ok" and len(r.results) == 5
    assert sum("Horror" in x["genres"] for x in r.results) >= 4
    assert r.results[0]["subtitle_score"] is not None and "best_chunk" in r.results[0]


def test_negation_pushes_genre_down(searcher):
    plain = searcher.search("funny movie with ghosts", top_k=8)
    neg = searcher.search("funny movie but no ghosts", top_k=8)
    assert neg.query["neg_terms"] == ["ghosts"]
    n_h = lambda r: sum("Horror" in x["genres"] for x in r.results)  # noqa: E731
    assert n_h(neg) <= n_h(plain)
    assert any(x["neg_penalty"] > 0 for x in plain.results + neg.results) or n_h(neg) == 0


def test_meta_negation_excludes_genre(built):
    d, _, _ = built
    s = SemanticSearcher.from_dir(d / "ix", SearchConfig(meta_negation=True, meta_neg_penalty=50.0))
    r = s.search("something scary but not horror, a story of a haunted house", top_k=10)
    assert all("Horror" not in x["genres"] for x in r.results[:5])


def test_user_input_never_raises(searcher):
    for q in ["", "   ", None, 5, "🙂", "asdfgh", "no", "not", "nothing but ,", "x" * 5000, "a\x00b", "NOT NOT NOT"]:
        r = searcher.search(q)
        assert r.status in {"ok", "ok_with_warning", "empty", "negation_only", "low_confidence", "error"}
        assert r.status != "error", q
    assert searcher.search("").results == []
    assert searcher.search("Nothing too scary and nothing too sad.").status == "negation_only"


def test_abstention(built):
    d, _, _ = built
    s = SemanticSearcher.from_dir(d / "ix", SearchConfig(min_confidence=0.99))
    r = s.search("a haunted house with ghosts")
    assert r.status == "low_confidence" and r.results == []
    assert len(s.search("a haunted house with ghosts", return_all=True).results) == 5


def test_exclude_id_and_top_k_validation(searcher):
    first = searcher.search("a haunted house with ghosts").results[0]["movie_id"]
    again = searcher.search("a haunted house with ghosts", exclude_imdb_id=first)
    assert first not in [x["movie_id"] for x in again.results]
    with pytest.raises(SemanticV2Error):
        searcher.search("x", top_k=0)


def test_whole_vs_clause_scores_are_finite(searcher):
    r = searcher.search("funny and silly", top_k=3)
    assert r.query["clauses"] == ["funny", "silly"]
    assert all(np.isfinite(x["score"]) for x in r.results)


def test_topk_mean_aggregation():
    sh = ChunkShard(emb=np.zeros((6, 2), np.float16), movie_local=np.array([0, 0, 0, 1, 1, 3], np.int32), m0=0, n_movies=4)
    scores = np.array([[1.0], [3.0], [2.0], [5.0], [1.0], [4.0]], np.float32)
    out = sh.topk_mean(scores, 2)[:, 0]
    assert out[0] == pytest.approx(2.5) and out[1] == pytest.approx(3.0) and np.isnan(out[2]) and out[3] == pytest.approx(4.0)


def test_hubness():
    s = np.array([[0.9, 0.1, 0.5], [np.nan] * 3, [0.2, 0.2, 0.2]], np.float32)
    h = hubness(s, 2)
    assert h[0] == pytest.approx(0.7) and np.isnan(h[1]) and h[2] == pytest.approx(0.2)


def test_resume_skips_finished_shards(built, tmp_path):
    d, cfg, _ = built
    out = tmp_path / "ix"
    shutil.copytree(d / "ix", out)
    mp = json.loads((out / "manifest.json").read_text())
    mp["complete"] = False                                   # pretend the run died after the shards
    (out / "manifest.json").write_text(json.dumps(mp))
    (out / "hub_offsets.npz").unlink()
    (out / "shards" / "shard_00002" / "DONE").unlink()       # last shard unfinished
    marker = out / "shards" / "shard_00000" / "chunk_emb.npy"
    before = marker.stat().st_mtime_ns
    m = build_index(d / "data.csv", out, cfg)
    assert m["complete"] and marker.stat().st_mtime_ns == before          # finished shard untouched
    assert (out / "shards" / "shard_00002" / "DONE").is_file() and (out / "hub_offsets.npz").is_file()


def test_build_refuses_overwrite_and_config_mismatch(built):
    d, cfg, _ = built
    with pytest.raises(SemanticV2Error):
        build_index(d / "data.csv", d / "ix", cfg)
    other = BuildConfig(field_model="debug-hashing", chunk_model="debug-hashing", chunk_words=61, shard_movies=10)
    with pytest.raises(SemanticV2Error):
        build_index(d / "data.csv", d / "ix", other)


def test_config_validation():
    with pytest.raises(SemanticV2Error):
        BuildConfig(chunk_model="minilm-l12", chunk_words=150).validate()       # 128-token window
    with pytest.raises(SemanticV2Error):
        BuildConfig(field_model="nope").validate()
    with pytest.raises(SemanticV2Error):
        SearchConfig(weights={"plot": 0.0}).validate()


def test_loader_streams_and_excludes(built):
    d, cfg, _ = built
    st = LoadStats()
    ids = [r.imdb_id for r in iter_movies(d / "data.csv", cfg, st)]
    assert len(ids) == len(set(ids)) == 24 and st.excluded_no_text == 1
