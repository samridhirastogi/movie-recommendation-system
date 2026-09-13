"""Offline evaluation: original notebook model vs. the improved model.

    python evaluate.py [--tune]

Ground truth is franchise membership derived only from titles
("Toy Story" / "Toy Story 2" / "Toy Story 3"), which the models never see.
For each franchise film we ask: how many of its siblings appear in the top 10?
We also report the average weighted rating of what gets recommended.

With --tune, a small grid search runs on half of the franchises and the
chosen settings are reported on the other, held-out half.
"""
from __future__ import annotations

import itertools
import re
import sys
import zlib

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from build_model import DEFAULT_WEIGHTS, build_blocks, combine, load_movies

K = 10
DEFAULT_QUALITY_WEIGHT = 0.25
ROMAN = {"ii", "iii", "iv", "v", "vi", "vii"}
# Keys containing these ("House of ...", "Rise of the ...") group unrelated films.
FILLER = {"of", "the", "a", "an", "in", "to", "who", "about", "and", "for", "on", "at", "with", "from",
          "is", "my", "i", "you", "all", "it's", "she's", "he's", "beyond", "what", "how", "when", "no",
          "not", "one", "man", "big", "little", "last", "love", "new", "our", "your", "we", "me", "be"}


def franchise_key(title: str) -> str:
    t = title.lower().replace("-", " ").split(":")[0]
    t = re.sub(r"^(the|a|an) ", "", t)
    words = re.findall(r"[a-z0-9']+", t)
    while words and (words[-1].isdigit() or words[-1] in ROMAN or words[-1] == "part"):
        words.pop()
    key = words[:2]
    return " ".join(key) if len(key) == 2 and not FILLER.intersection(key) else ""


def franchise_groups(movies: pd.DataFrame) -> list[np.ndarray]:
    keys = movies["title"].apply(franchise_key)
    groups = [np.array(ix) for key, ix in movies.groupby(keys).indices.items()
              if key and 2 <= len(ix) <= 10]
    return groups


def baseline_features(movies: pd.DataFrame):
    """Re-implementation of the original notebook pipeline."""
    ps = PorterStemmer()
    squash = lambda xs: [x.replace(" ", "") for x in xs]
    tags = (movies["overview"].str.split()
            + movies["genres"].apply(squash) + movies["keywords"].apply(squash)
            + movies["cast"].apply(lambda c: squash(c[:3])) + movies["directors"].apply(lambda d: squash(d[:1])))
    tags = tags.apply(lambda t: " ".join(ps.stem(w) for w in " ".join(t).lower().split()))
    x = CountVectorizer(max_features=5000, stop_words="english").fit_transform(tags)
    return normalize(x.astype(np.float32))


def top_k(features, quality: np.ndarray | None, quality_weight: float) -> np.ndarray:
    sim = (features @ features.T).toarray()
    np.fill_diagonal(sim, -np.inf)
    if quality is not None:
        sim = sim * ((1 - quality_weight) + quality_weight * quality)[None, :]
    return np.argpartition(-sim, K, axis=1)[:, :K]


def evaluate(recs: np.ndarray, groups: list[np.ndarray], movies: pd.DataFrame) -> dict[str, float]:
    recall, hits = [], []
    for g in groups:
        members = set(g)
        for i in g:
            found = len(members.intersection(recs[i]))
            recall.append(found / min(K, len(g) - 1))
            hits.append(found > 0)
    rating = movies["weighted_rating"].to_numpy()[recs].mean()
    obscure = (movies["vote_count"].to_numpy()[recs] < 50).mean()
    return {
        "franchise_recall@10": np.mean(recall),
        "franchise_hit@10": np.mean(hits),
        "avg_weighted_rating": rating,
        "share_obscure(<50 votes)": obscure,
    }


def tune(blocks, movies, groups):
    grid = {
        "text": [1.0],
        "keywords": [0.5, 0.8],
        "genres": [0.3, 0.6],
        "cast": [0.4, 0.6, 0.8],
        "directors": [0.4, 0.7, 1.0],
        "writers": [0.0, 0.5],
    }
    best = None
    for values in itertools.product(*grid.values()):
        weights = dict(zip(grid, values))
        recs = top_k(combine(blocks, weights), None, 0)
        score = evaluate(recs, groups, movies)["franchise_recall@10"]
        if best is None or score > best[0]:
            best = (score, weights)
    print(f"best tuning recall@10={best[0]:.3f} with {best[1]}")
    return best[1]


def main() -> None:
    movies = load_movies()
    groups = franchise_groups(movies)
    tune_groups = [g for g in groups if zlib.crc32(movies["title"][g[0]].encode()) % 2 == 0]
    test_groups = [g for g in groups if zlib.crc32(movies["title"][g[0]].encode()) % 2 == 1]
    print(f"{len(groups)} franchises ({sum(map(len, groups))} films); "
          f"e.g. {[movies['title'][g].tolist() for g in groups[:3]]}")

    blocks = build_blocks(movies)
    weights = tune(blocks, movies, tune_groups) if "--tune" in sys.argv else DEFAULT_WEIGHTS
    report_groups = test_groups if "--tune" in sys.argv else groups

    rows = {
        "original (notebook)": evaluate(top_k(baseline_features(movies), None, 0), report_groups, movies),
        "improved, content only": evaluate(top_k(combine(blocks, weights), None, 0), report_groups, movies),
        f"improved + quality ({DEFAULT_QUALITY_WEIGHT})": evaluate(
            top_k(combine(blocks, weights), movies["quality"].to_numpy(), DEFAULT_QUALITY_WEIGHT),
            report_groups, movies),
    }
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
