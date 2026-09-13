import hashlib
import html
import os

import requests
import streamlit as st

from recommender import Recommender

st.set_page_config(page_title="CineMatch · Movie Recommender", page_icon="🎬", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.hero h1 {font-size: 2.6rem; margin-bottom: 0; letter-spacing: -0.02em;}
.hero p {color: #9aa3b2; margin-top: .25rem; font-size: 1.05rem;}
.card {background: #161b26; border: 1px solid #252c3b; border-radius: 14px; overflow: hidden; margin-bottom: 1rem;}
.poster {aspect-ratio: 2 / 3; width: 100%; object-fit: cover; display: block;}
.banner {height: 84px; display: flex; align-items: center; justify-content: space-between; padding: 0 .9rem;}
.banner .emoji {font-size: 2.1rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,.35));}
.banner .year {color: rgba(255,255,255,.85); font-weight: 700; font-size: .95rem;}
.card-body {padding: .75rem .85rem .85rem; display: flex; flex-direction: column; gap: .35rem;}
.card-title {font-weight: 650; font-size: 1rem; line-height: 1.25; color: #f1f3f7;}
.meta {color: #9aa3b2; font-size: .82rem;}
.row {display: flex; justify-content: space-between; align-items: center; gap: .4rem; flex-wrap: wrap;}
.rating {color: #f5c518; font-weight: 600; font-size: .9rem; white-space: nowrap;}
.match {background: #e50914; color: #fff; border-radius: 999px; padding: 1px 8px; font-size: .75rem;
        font-weight: 600; white-space: nowrap;}
.pill {display: inline-block; background: #232a39; color: #c8cfdb; border-radius: 999px;
       padding: 1px 8px; margin: 0 4px 4px 0; font-size: .72rem;}
.why {color: #9aa3b2; font-size: .78rem; line-height: 1.4; border-top: 1px solid #252c3b; padding-top: .45rem;}
.why b {color: #d7dce5; font-weight: 600;}
.card details summary {cursor: pointer; color: #c8cfdb; font-size: .8rem; font-weight: 600;}
.card details p {color: #b3bac6; font-size: .8rem; line-height: 1.45; margin: .4rem 0 0;}
.card details a {color: #ff5a63;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🧭", "Animation": "🎨", "Comedy": "😂", "Crime": "🕵️",
    "Documentary": "🎥", "Drama": "🎭", "Family": "🏡", "Fantasy": "🐉", "Foreign": "🌍",
    "History": "📜", "Horror": "👻", "Music": "🎵", "Mystery": "🔍", "Romance": "❤️",
    "Science Fiction": "🚀", "TV Movie": "📺", "Thriller": "🔪", "War": "🎖️", "Western": "🤠",
}


@st.cache_resource
def load_recommender() -> Recommender:
    return Recommender.load()


def tmdb_key() -> str | None:
    try:
        return st.secrets.get("TMDB_API_KEY") or os.getenv("TMDB_API_KEY")
    except FileNotFoundError:
        return os.getenv("TMDB_API_KEY")


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def poster_url(movie_id: int, api_key: str) -> str | None:
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key}, timeout=4)
        path = r.json().get("poster_path") if r.ok else None
        return f"https://image.tmdb.org/t/p/w342{path}" if path else None
    except requests.RequestException:
        return None


def banner(movie, year: str) -> str:
    """Compact coloured header used when no poster is available."""
    hue = int(hashlib.md5(movie["title"].encode()).hexdigest()[:2], 16) * 360 // 256
    style = f"background: linear-gradient(120deg, hsl({hue},55%,40%), hsl({(hue + 40) % 360},60%,16%));"
    emoji = GENRE_EMOJI.get(movie["genres"][0], "🎬") if movie["genres"] else "🎬"
    return f'<div class="banner" style="{style}"><span class="emoji">{emoji}</span><span class="year">{year}</span></div>'


def card(movie, rec: Recommender, seed_ids: list[int] | None, api_key: str | None) -> str:
    year = "" if str(movie["year"]) in {"<NA>", "None"} else str(movie["year"])
    url = poster_url(int(movie["id"]), api_key) if api_key else None
    image = f'<img class="poster" src="{html.escape(url)}" alt="">' if url else banner(movie, year)
    runtime = f"{int(movie['runtime'])} min" if movie["runtime"] == movie["runtime"] and movie["runtime"] else ""
    meta = " · ".join(x for x in [year, runtime, ", ".join(movie["directors"][:1])] if x)
    # Cosine similarities sit around 0.1-0.6 for good matches; sqrt spreads them over a readable range.
    badge = f'<span class="match">{movie["similarity"] ** 0.5 * 100:.0f}% match</span>' if "similarity" in movie else ""
    pills = "".join(f'<span class="pill">{html.escape(g)}</span>' for g in movie["genres"][:3])

    why = ""
    if seed_ids:
        shared = rec.explain(seed_ids, movie)
        parts = []
        if shared["directors"]:
            parts.append(f"Same director <b>{html.escape(', '.join(shared['directors']))}</b>")
        if shared["cast"]:
            parts.append(f"Starring <b>{html.escape(', '.join(shared['cast'][:2]))}</b>")
        if shared["keywords"]:
            parts.append("Themes: " + html.escape(", ".join(shared["keywords"][:3])))
        elif shared["genres"]:
            parts.append("Also " + html.escape(", ".join(shared["genres"][:2])))
        why = f'<div class="why">{"<br>".join(parts)}</div>' if parts else ""

    extra = ""
    if movie["cast"]:
        extra += f'<p>Cast: {html.escape(", ".join(movie["cast"]))}</p>'
    if movie["homepage"]:
        extra += f'<p><a href="{html.escape(movie["homepage"])}" target="_blank" rel="noopener">Official site ↗</a></p>'
    details = (f'<details><summary>Overview</summary><p>{html.escape(movie["overview"] or "No overview available.")}</p>'
               f"{extra}</details>")

    # No leading indentation: Markdown would treat indented lines as a code block.
    return "".join([
        f'<div class="card">{image}<div class="card-body">',
        f'<div class="card-title">{html.escape(movie["title"])}</div>',
        f'<div class="meta">{html.escape(meta)}</div>',
        f'<div class="row"><span class="rating">★ {movie["vote_average"]:.1f} ',
        f'<span class="meta">({int(movie["vote_count"]):,})</span></span>{badge}</div>',
        f"<div>{pills}</div>{why}{details}",
        "</div></div>",
    ])


def grid(movies, rec: Recommender, seed_ids: list[int] | None, api_key: str | None, per_row: int = 5):
    rows = [movie for _, movie in movies.iterrows()]
    for start in range(0, len(rows), per_row):
        for col, movie in zip(st.columns(per_row), rows[start:start + per_row]):
            col.markdown(card(movie, rec, seed_ids, api_key), unsafe_allow_html=True)


rec = load_recommender()
movies = rec.movies
api_key = tmdb_key()

labels = {
    int(r.id): f"{r.title} ({r.year})" if str(r.year) != "<NA>" else r.title
    for r in movies.sort_values("popularity", ascending=False).itertuples()
}

st.markdown('<div class="hero"><h1>🎬 CineMatch</h1>'
            "<p>Pick movies you love and get recommendations based on story, themes, cast and crew.</p></div>",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    all_genres = sorted({g for gs in movies["genres"] for g in gs})
    genres = st.multiselect("Genres", all_genres)
    years = movies["year"].dropna().astype(int)
    year_range = st.slider("Release year", int(years.min()), int(years.max()), (1970, int(years.max())))
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, 0.5)
    hide_obscure = st.toggle("Hide films with fewer than 50 votes", value=True)
    st.header("Ranking")
    count = st.select_slider("Number of recommendations", [5, 10, 15, 20], value=10)
    quality_weight = st.slider("Favour highly-rated films", 0.0, 1.0, 0.25, 0.05,
                               help="0 = pure similarity; higher values boost films with a strong weighted rating.")
    if not api_key:
        st.caption("Tip: add a `TMDB_API_KEY` secret to show movie posters.")

# Selections live in the URL (?movie=155&movie=27205) so results can be shared.
if "seeds" not in st.session_state:
    from_url = [int(x) for x in st.query_params.get_all("movie") if x.isdigit()]
    st.session_state.seeds = [i for i in from_url if i in labels][:5]
seed_ids = st.multiselect(
    "Movies you like",
    options=list(labels),
    format_func=labels.get,
    placeholder="Search for a movie, e.g. Inception",
    max_selections=5,
    key="seeds",
)
st.query_params["movie"] = [str(i) for i in seed_ids]

filters = dict(genres=genres or None, year_range=year_range, min_rating=min_rating,
               min_votes=50 if hide_obscure else 0)

if seed_ids:
    results = rec.recommend(seed_ids, k=count, quality_weight=quality_weight, **filters)
    picked = " + ".join(movies.loc[movies["id"].isin(seed_ids), "title"])
    st.subheader(f"Because you like {picked}")
    if results.empty:
        st.info("No movies match these filters. Try widening the year range or lowering the minimum rating.")
    else:
        grid(results, rec, seed_ids, api_key)
else:
    st.subheader("Top rated to get you started")
    top = movies[movies["vote_count"] >= filters["min_votes"]]
    top = top[(top["year"].fillna(0) >= year_range[0]) & (top["year"].fillna(0) <= year_range[1])
              & (top["vote_average"] >= min_rating)]
    if genres:
        top = top[top["genres"].apply(lambda g: bool(set(genres) & set(g)))]
    grid(top.nlargest(count, "weighted_rating"), rec, None, api_key)

st.caption("Data: TMDB 5000 Movie Dataset. This product uses the TMDB API but is not endorsed or certified by TMDB.")
