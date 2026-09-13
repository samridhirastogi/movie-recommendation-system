"""Build recommender artifacts from the raw TMDB 5000 CSVs.

    python build_model.py

Improvements over the original notebook:
  * movies and credits are joined on the TMDB id (joining on title duplicated
    rows for "Batman", "The Host" and "Out of the Blue");
  * each field is vectorised separately with TF-IDF, so rare, informative
    terms (a director, a niche keyword) outweigh common ones ("drama");
  * fields are blended with tuned weights instead of one bag of words;
  * a Bayesian weighted rating is stored so ranking can favour good films;
  * a sparse feature matrix (a few MB) replaces the 185 MB dense similarity matrix.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

# Tuned with `python evaluate.py --tune` (see README for the results).
DEFAULT_WEIGHTS = {
    "text": 1.0,
    "keywords": 0.5,
    "genres": 0.3,
    "cast": 0.4,
    "directors": 0.4,
    "writers": 0.5,
}
WRITER_JOBS = {"Screenplay", "Writer", "Novel", "Story", "Author", "Characters"}

_stemmer = SnowballStemmer("english")
_word_re = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _names(raw: str, limit: int | None = None) -> list[str]:
    items = [d["name"] for d in json.loads(raw)] if isinstance(raw, str) else []
    return list(dict.fromkeys(items))[:limit]


def _crew(raw: str, jobs: set[str], limit: int) -> list[str]:
    items = [d["name"] for d in json.loads(raw) if d["job"] in jobs] if isinstance(raw, str) else []
    return list(dict.fromkeys(items))[:limit]


def load_movies() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "tmdb_5000_movies.csv.zip")
    credits = pd.read_csv(ROOT / "tmdb_5000_credits.csv.zip")
    df = raw.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id", how="left")

    movies = pd.DataFrame({
        "id": df["id"],
        "title": df["title"],
        "overview": df["overview"].fillna(""),
        "tagline": df["tagline"].fillna(""),
        "genres": df["genres"].apply(_names),
        "keywords": df["keywords"].apply(_names),
        "cast": df["cast"].apply(_names, limit=5),
        "directors": df["crew"].apply(_crew, jobs={"Director"}, limit=2),
        "writers": df["crew"].apply(_crew, jobs=WRITER_JOBS, limit=4),
        "year": pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype("Int64"),
        "runtime": df["runtime"],
        "language": df["original_language"],
        "vote_average": df["vote_average"],
        "vote_count": df["vote_count"],
        "popularity": df["popularity"],
        "homepage": df["homepage"].fillna(""),
    })

    # IMDb-style Bayesian average: films with few votes shrink towards the mean.
    v, r = movies["vote_count"], movies["vote_average"]
    m = v.quantile(0.5)
    c = (r * v).sum() / v.sum()
    weighted = v / (v + m) * r + m / (v + m) * c
    movies["weighted_rating"] = weighted.round(2)
    movies["quality"] = ((weighted - weighted.min()) / (weighted.max() - weighted.min())).round(4)
    return movies


def _text_tokens(text: str) -> list[str]:
    words = [_stemmer.stem(w) for w in _word_re.findall(text.lower())
             if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def _entity_tokens(values: list[str]) -> list[str]:
    return [re.sub(r"\W+", "", v.lower()) for v in values]


def build_blocks(movies: pd.DataFrame) -> dict[str, sp.csr_matrix]:
    """One row-normalised TF-IDF matrix per field."""
    text = movies["overview"] + " " + movies["tagline"] + " " + movies["keywords"].str.join(" ")
    blocks = {
        "text": TfidfVectorizer(analyzer=_text_tokens, sublinear_tf=True, min_df=2, max_df=0.5).fit_transform(text)
    }
    for field in ["keywords", "genres", "cast", "directors", "writers"]:
        vec = TfidfVectorizer(analyzer=_entity_tokens, min_df=2 if field != "genres" else 1)
        blocks[field] = vec.fit_transform(movies[field])
    return {name: normalize(x).tocsr() for name, x in blocks.items()}


def combine(blocks: dict[str, sp.csr_matrix], weights: dict[str, float]) -> sp.csr_matrix:
    parts = [blocks[name] * w for name, w in weights.items() if w > 0]
    return normalize(sp.hstack(parts).tocsr()).astype(np.float32)


def main() -> None:
    movies = load_movies()
    features = combine(build_blocks(movies), DEFAULT_WEIGHTS)

    ARTIFACTS.mkdir(exist_ok=True)
    movies.drop(columns=["tagline"]).to_parquet(ARTIFACTS / "movies.parquet", index=False)
    sp.save_npz(ARTIFACTS / "features.npz", features)
    print(f"{len(movies)} movies, {features.shape[1]} features, {features.nnz} non-zeros -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
