"""Minimal movie recommender app (Streamlit) on top of Semantic Search v2.

Run from the repository root (or double-click start_app.bat):
    py -m pip install -r requirements-app.txt
    py -m streamlit run app.py

Optional env vars:
    MOVIE_INDEX       index folder       (default: models/v2_sample_bge)
    MOVIE_CONFIG      config JSON        (default: configs/v2_default.json)
    MOVIE_DATA        movie data file, CSV or Parquet (default: data/processed/sample_clean_movies.csv)
    HF_HUB_OFFLINE=1  skip HuggingFace network calls after the first run

Posters and plots come from the movie data file (columns imdb_id, poster_url, clean_plot).
On the first start the app copies just those columns into app_details.csv (a few MB),
and from then on reads only that small file, so the big data file is no longer needed.
Delete app_details.csv whenever you switch to a different data file.
Without either file the app still works: short plot snippet from the index, no posters.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.semantic_v2.config import SearchConfig
from src.semantic_v2.search import SemanticSearcher

INDEX_DIR = os.environ.get("MOVIE_INDEX", "models/v2_bgebase_bgesmall")
CONFIG_PATH = os.environ.get("MOVIE_CONFIG", "configs/v2_default.json")
MOVIE_DATA = os.environ.get("MOVIE_DATA", "data/processed/clean_movies.parquet")  # .csv or .parquet
DETAILS_CSV = "app_details.csv"  # small cache: imdb_id, poster_url, clean_plot
DETAIL_COLS = ["imdb_id", "poster_url", "clean_plot"]
FIRST_N = 3   # movies shown right after a search
STEP_N = 3    # added by each "Show more" click
MAX_N = 10    # upper limit (also how many the search returns)


def imdb_tt(movie_id) -> str:
    """'111161', '0111161' and 'tt0111161' all become 'tt0111161'."""
    s = str(movie_id).strip()
    if s.endswith(".0"):  # ids read as floats
        s = s[:-2]
    return s if s.startswith("tt") else "tt" + s.zfill(7)


def fix_url(url) -> str | None:
    """Repairs common URL glitches; returns None for empty values."""
    u = str(url).strip()
    if not u or u.lower() in ("nan", "none"):
        return None
    if u.startswith("https//"):
        u = "https://" + u[len("https//"):]
    elif u.startswith("http//"):
        u = "http://" + u[len("http//"):]
    elif u.startswith("/"):  # bare TMDB poster_path like /abc.jpg
        u = "https://image.tmdb.org/t/p/w500" + u
    elif not u.startswith("http"):
        u = "https://" + u
    return u


@st.cache_resource(show_spinner="Loading the model (first start takes a while)...")
def load_searcher():
    # Runs once per server process, not on every click.
    return SemanticSearcher.from_dir(INDEX_DIR, SearchConfig.from_json(CONFIG_PATH))


@st.cache_resource
def plot_lookup(_searcher):
    """imdb_id -> short plot snippet from the index metadata (no extra files needed)."""
    movies = _searcher.index.movies
    if "plot_snippet" not in movies.columns:
        return {}
    return dict(zip(movies["imdb_id"].map(imdb_tt), movies["plot_snippet"].astype(str)))


def read_details(path: Path) -> pd.DataFrame:
    """Reads only the needed columns from a CSV or Parquet file (the full data also holds subtitles)."""
    if path.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        names = pq.read_schema(path).names
        cols = [c for c in DETAIL_COLS if c in names]
        if "imdb_id" not in cols:
            return pd.DataFrame()
        # Parquet reads just these columns from disk, so this is fast even for a multi-GB file.
        return pd.read_parquet(path, columns=cols).fillna("").astype(str)
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                             usecols=lambda c: c in DETAIL_COLS)
        except Exception:
            continue
        if "imdb_id" in df.columns:
            return df
    return pd.DataFrame()


@st.cache_resource(show_spinner="Loading movie details...")
def details_lookup():
    """imdb_id -> {'poster': url, 'plot': text}; empty dict if no data file is found."""
    small, big = Path(DETAILS_CSV), Path(MOVIE_DATA)
    if small.is_file():
        df = read_details(small)
    elif big.is_file():
        df = read_details(big)
        if df.empty or "poster_url" not in df.columns:
            st.warning(f"{big}: no imdb_id / poster_url columns found, so no posters are shown.")
        if not df.empty:
            df.to_csv(small, index=False)  # next start reads only this small file
    else:
        st.warning(f"Movie data file not found: {big} - showing results without posters.")
        return {}
    if df.empty:
        return {}
    out = {}
    for row in df.itertuples(index=False):
        row = row._asdict()
        out[imdb_tt(row["imdb_id"])] = {
            "poster": fix_url(row.get("poster_url", "")),
            "plot": (row.get("clean_plot") or "").strip() or None,
        }
    return out


def clean_genres(raw: str) -> str:
    return str(raw).replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()


st.set_page_config(page_title="Movie mood picker", page_icon="🎬")
st.title("🎬 Movie mood picker")
st.caption("Describe how you feel or what you'd like to watch.")

searcher = load_searcher()
plots = plot_lookup(searcher)
details = details_lookup()

with st.form("search"):
    query = st.text_input("Your mood", max_chars=100,
                          placeholder="something cozy and a little bit grim")
    submitted = st.form_submit_button("Find movies")

# Results are kept in session_state, so clicking "Show more" (which reruns the script)
# does not lose them or repeat the search.
if submitted:
    if not query.strip():
        st.session_state.pop("response", None)
        st.warning("Type something first.")
    else:
        with st.spinner("Searching..."):
            st.session_state.response = searcher.search(query.strip(), top_k=MAX_N, snippets=False)
        st.session_state.shown = FIRST_N

r = st.session_state.get("response")
if r is not None:
    if r.status == "error":
        st.error(r.message)
    elif r.message:
        st.info(r.message)
    for w in r.warnings or []:  # warnings are dicts: {"code": ..., "message": ...}
        st.warning(w.get("message", str(w)) if isinstance(w, dict) else str(w))

    results = r.results or []
    shown = min(st.session_state.get("shown", FIRST_N), len(results))
    for m in results[:shown]:
        tt = imdb_tt(m["movie_id"])
        info = details.get(tt, {})
        snippet = plots.get(tt, "")
        description = info.get("plot") or (snippet + "…" if snippet else "")

        st.divider()
        img_col, text_col = st.columns([1, 2.5])
        with img_col:
            if info.get("poster"):
                st.image(info["poster"])
        with text_col:
            year = m.get("year")
            st.subheader(f"{m['rank']}. {m['title']}" + (f" ({year})" if year else ""))
            meta = [clean_genres(m.get("genres", "")), f"match score {m['score']:.2f}"]
            st.caption(" · ".join(x for x in meta if x))
            if description:
                st.write(description)
            st.markdown(f"[IMDb](https://www.imdb.com/title/{tt}/)")

    if shown < len(results):
        st.divider()
        if st.button(f"Show more ({shown} of {len(results)} shown)"):
            st.session_state.shown = shown + STEP_N
            st.rerun()
