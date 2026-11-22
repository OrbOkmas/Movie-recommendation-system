"""Build (sharded, resumable) and load the v2 index.

Layout of ``<index_dir>``::

    manifest.json            config, statistics, fingerprint, COMPLETE flag
    shards/shard_00000/      movies.csv, plot.npy, keywords.npy, genres.npy (float16, zero rows = missing),
                             chunk_emb.npy, chunk_movie.npy (local movie ids), chunk_text.parquet, DONE
    hub_offsets.npz          per-source "hubness" of every movie (NaN = source missing)
    hub_bank.txt             background queries used for the hub correction
    vocab.tsv                domain word counts for the spell checker

A shard is the unit of resumable work: if encoding is interrupted, ``train`` without ``--overwrite`` continues
with the first shard that has no DONE marker. Shards can be copied between machines.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .chunks import ChunkShard
from .config import FIELDS, BuildConfig, SemanticV2Error
from .encode import TextEncoder, load_encoders
from .fields import LoadStats, MovieRecord, iter_movies, vocab_counts
from .lexicon import BANK_MOODS, BANK_TEMPLATES_ADJ2, BANK_TEMPLATES_THEME, BANK_THEMES

LOGGER = logging.getLogger("semantic_v2")
INDEX_VERSION = 2
FIELD_ATTR = {"plot": "plot_text", "keywords": "keywords_text", "genres": "genres_text"}


# ----------------------------------------------------------------------------- background bank
def build_bank(plot_sentences: list[str], cfg: BuildConfig, exclude: Iterable[str] = ()) -> list[str]:
    """Generic background queries (templated mood phrases + random plot sentences).

    Never built from evaluation queries; ``exclude`` removes any accidental overlap with them.
    """
    rng = random.Random(cfg.seed)
    banned = {" ".join(x.lower().split()) for x in exclude}
    bank: list[str] = []
    seen: set[str] = set()

    def add(t: str):
        key = " ".join(t.lower().split())
        if key and key not in seen and key not in banned:
            seen.add(key)
            bank.append(t)

    guard = 0
    while len(bank) < cfg.hub_bank_templated and guard < cfg.hub_bank_templated * 20:
        guard += 1
        if rng.random() < 0.5:
            a, b = rng.sample(BANK_MOODS, 2)
            add(rng.choice(BANK_TEMPLATES_ADJ2).format(a=a, b=b))
        else:
            add(rng.choice(BANK_TEMPLATES_THEME).format(a=rng.choice(BANK_MOODS), t=rng.choice(BANK_THEMES)))
    for s in plot_sentences[:cfg.hub_bank_plot_sentences]:
        add(s)
    return bank


def hubness(scores: np.ndarray, k: int) -> np.ndarray:
    """scores (n_movies, n_bank), NaN rows allowed -> mean of the k highest bank scores per movie."""
    n_bank = scores.shape[1]
    kk = min(k, n_bank)
    out = np.full(scores.shape[0], np.nan, dtype=np.float32)
    ok = ~np.isnan(scores).all(axis=1)
    if ok.any():
        top = -np.partition(-np.nan_to_num(scores[ok], nan=-np.inf), kk - 1, axis=1)[:, :kk]
        out[ok] = top.mean(axis=1)
    return out


# ----------------------------------------------------------------------------- build
def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:16]


def _encode_shard(records: list[MovieRecord], fe: TextEncoder, ce: TextEncoder, shard_dir: Path,
                  store_chunk_text: bool) -> dict:
    t0 = time.time()
    n = len(records)
    shard_dir.mkdir(parents=True, exist_ok=True)
    flags = {}
    for name, attr in FIELD_ATTR.items():
        idx = [i for i, r in enumerate(records) if getattr(r, attr)]
        arr = np.zeros((n, fe.dim), dtype=np.float16)
        if idx:
            arr[idx] = fe.encode_docs([getattr(records[i], attr) for i in idx]).astype(np.float16)
        np.save(shard_dir / f"{name}.npy", arr)
        flags[f"has_{name}"] = [bool(getattr(r, attr)) for r in records]
    texts, movie_local = [], []
    for i, r in enumerate(records):
        texts.extend(r.chunks)
        movie_local.extend([i] * len(r.chunks))
    chunk_emb = ce.encode_docs(texts).astype(np.float16) if texts else np.zeros((0, ce.dim), np.float16)
    np.save(shard_dir / "chunk_emb.npy", chunk_emb)
    np.save(shard_dir / "chunk_movie.npy", np.asarray(movie_local, dtype=np.int32))
    if store_chunk_text and texts:
        pd.DataFrame({"chunk_text": texts}).to_parquet(shard_dir / "chunk_text.parquet", index=False)
    meta = pd.DataFrame({"imdb_id": [r.imdb_id for r in records], "title": [r.title for r in records],
                         "year": [r.year for r in records], "genres": [r.genres_raw for r in records],
                         "keywords": [r.keywords_raw for r in records],
                         "plot_snippet": [" ".join(r.plot.split()[:45]) for r in records],
                         "n_chunks": [len(r.chunks) for r in records], **flags})
    meta.to_csv(shard_dir / "movies.csv", index=False, encoding="utf-8")
    info = {"n_movies": n, "n_chunks": len(texts), "ids_hash": _ids_hash([r.imdb_id for r in records]),
            "seconds": round(time.time() - t0, 1)}
    (shard_dir / "DONE").write_text(json.dumps(info), encoding="utf-8")
    return info


def build_index(data_path: str | Path, out_dir: str | Path, cfg: BuildConfig, *, max_rows: int | None = None,
                overwrite: bool = False, store_chunk_text: bool = True, device: str | None = None,
                exclude_texts: Iterable[str] = ()) -> dict:
    cfg.validate()
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if out_dir.exists() and any(out_dir.iterdir()):
        if overwrite:
            shutil.rmtree(out_dir)
        elif manifest_path.is_file():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != cfg.fingerprint():
                raise SemanticV2Error("Index directory has a different build config; use --overwrite or another dir.")
            if old.get("complete"):
                raise SemanticV2Error(f"{out_dir} is already complete. Use --overwrite to rebuild.")
            LOGGER.info("Resuming the unfinished build in %s", out_dir)
        else:
            raise SemanticV2Error(f"{out_dir} is not empty and has no manifest; use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"index_version": INDEX_VERSION, "fingerprint": cfg.fingerprint(), "config": asdict(cfg),
                "complete": False, "data": str(data_path), "max_rows": max_rows}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    fe, ce = load_encoders(cfg.field_model, cfg.chunk_model, batch_size=cfg.batch_size, device=device)
    stats, vocab = LoadStats(), {}
    rng = random.Random(cfg.seed)
    reservoir: list[str] = []
    seen_plots = 0
    t0 = time.time()
    shard_infos: list[dict] = []
    buf: list[MovieRecord] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        sid = len(shard_infos)
        sdir = out_dir / "shards" / f"shard_{sid:05d}"
        done = sdir / "DONE"
        ids_hash = _ids_hash([r.imdb_id for r in buf])
        if done.is_file() and json.loads(done.read_text())["ids_hash"] == ids_hash:
            info = json.loads(done.read_text())
            LOGGER.info("shard %d: already done (%d movies) - skipped", sid, info["n_movies"])
        else:
            if sdir.exists():
                shutil.rmtree(sdir)
            info = _encode_shard(buf, fe, ce, sdir, store_chunk_text)
            LOGGER.info("shard %d: %d movies, %d chunks, %.0fs (total %.0fs)", sid, info["n_movies"],
                        info["n_chunks"], info["seconds"], time.time() - t0)
        shard_infos.append(info)
        buf = []

    for rec in iter_movies(data_path, cfg, stats, max_rows=max_rows):
        vocab_counts(rec, vocab)
        if rec.plot:  # reservoir sample of first plot sentences for the background bank
            first = rec.plot.split(". ")[0].strip()
            if 4 <= len(first.split()) <= 30:
                seen_plots += 1
                if len(reservoir) < cfg.hub_bank_plot_sentences:
                    reservoir.append(first)
                else:
                    j = rng.randrange(seen_plots)
                    if j < cfg.hub_bank_plot_sentences:
                        reservoir[j] = first
        buf.append(rec)
        if len(buf) >= cfg.shard_movies:
            flush()
    flush()
    if not shard_infos:
        raise SemanticV2Error("No rankable movies found in the data.")

    # vocabulary for the spell checker
    pd.Series(vocab).sort_values(ascending=False).to_csv(out_dir / "vocab.tsv", sep="\t", header=False)

    index = MovieIndex.load(out_dir, load_chunk_text=False, require_complete=False)
    bank = build_bank(reservoir, cfg, exclude_texts)
    (out_dir / "hub_bank.txt").write_text("\n".join(bank), encoding="utf-8")
    offsets = compute_hub_offsets(index, bank, fe, ce, cfg)
    np.savez(out_dir / "hub_offsets.npz", **offsets)

    manifest.update({"complete": True, "stats": {
        **asdict(stats), "indexed_movies": index.n_movies, "n_shards": len(shard_infos),
        "n_chunks": int(sum(s["n_chunks"] for s in shard_infos)), "hub_bank_size": len(bank),
        "field_dim": fe.dim, "chunk_dim": ce.dim, "build_seconds": round(time.time() - t0, 1)}})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Index complete: %s", manifest["stats"])
    return manifest


def compute_hub_offsets(index: "MovieIndex", bank: list[str], fe: TextEncoder, ce: TextEncoder,
                        cfg: BuildConfig) -> dict[str, np.ndarray]:
    bank_f = fe.encode_queries(bank)
    bank_c = bank_f if ce is fe else ce.encode_queries(bank)
    out: dict[str, np.ndarray] = {}
    for name in FIELDS:
        s = index.field_emb[name].astype(np.float32) @ bank_f.T
        s[~index.field_mask[name]] = np.nan
        out[name] = hubness(s, cfg.hub_k)
    sub = np.full(index.n_movies, np.nan, dtype=np.float32)
    for sh in index.shards:
        if sh.emb.shape[0] == 0:
            continue
        agg = sh.topk_mean(sh.raw_scores(bank_c), cfg.agg_topk)
        sub[sh.m0:sh.m0 + sh.n_movies] = hubness(agg, cfg.hub_k)
    out["subtitles"] = sub
    return out


# ----------------------------------------------------------------------------- load
class MovieIndex:
    """Read-only view of a built index."""

    def __init__(self):
        self.manifest: dict = {}
        self.cfg: BuildConfig = BuildConfig()
        self.movies: pd.DataFrame = pd.DataFrame()
        self.field_emb: dict[str, np.ndarray] = {}
        self.field_mask: dict[str, np.ndarray] = {}
        self.shards: list[ChunkShard] = []
        self.hub: dict[str, np.ndarray] = {}
        self.vocab: dict[str, int] = {}
        self.n_movies = 0
        self.dir = Path(".")

    @classmethod
    def load(cls, index_dir: str | Path, *, load_chunk_text: bool = True, require_complete: bool = True) -> "MovieIndex":
        d = Path(index_dir)
        mpath = d / "manifest.json"
        if not mpath.is_file():
            raise SemanticV2Error(f"No manifest.json in {d}; build the index first.")
        ix = cls()
        ix.dir = d
        ix.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if ix.manifest.get("index_version") != INDEX_VERSION:
            raise SemanticV2Error("Index was built by a different version of semantic_v2.")
        if require_complete and not ix.manifest.get("complete"):
            raise SemanticV2Error("Index build is not complete; re-run train to resume it.")
        ix.cfg = BuildConfig(**ix.manifest["config"])
        sdirs = sorted(p for p in (d / "shards").glob("shard_*") if (p / "DONE").is_file())
        metas, fields = [], {n: [] for n in FIELDS}
        m0 = 0
        for sd in sdirs:
            meta = pd.read_csv(sd / "movies.csv", dtype={"imdb_id": str, "year": str}, keep_default_na=False)
            metas.append(meta)
            for n in FIELDS:
                fields[n].append(np.load(sd / f"{n}.npy"))
            sh = ChunkShard(emb=np.load(sd / "chunk_emb.npy", mmap_mode="r"),
                            movie_local=np.load(sd / "chunk_movie.npy"), m0=m0, n_movies=len(meta),
                            text_path=(sd / "chunk_text.parquet") if load_chunk_text else None)
            ix.shards.append(sh)
            m0 += len(meta)
        ix.movies = pd.concat(metas, ignore_index=True)
        ix.n_movies = len(ix.movies)
        for n in FIELDS:
            ix.field_emb[n] = np.vstack(fields[n])
            ix.field_mask[n] = ix.movies[f"has_{n}"].to_numpy(dtype=bool)
        ix.field_mask["subtitles"] = ix.movies["n_chunks"].to_numpy() > 0
        hp = d / "hub_offsets.npz"
        if hp.is_file():
            z = np.load(hp)
            ix.hub = {k: z[k] for k in z.files}
        vp = d / "vocab.tsv"
        if vp.is_file():
            ix.vocab = {w: int(c) for w, c in pd.read_csv(vp, sep="\t", header=None, keep_default_na=False).itertuples(index=False)}
        return ix
