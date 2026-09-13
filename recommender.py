"""Hybrid movie recommender: loads precomputed artifacts and ranks movies.

Artifacts:
  artifacts/movies.parquet   one row per movie (metadata used for display/filters)  - build_model.py
  artifacts/features.npz     L2-normalised sparse content features (rows align with movies) - build_model.py
  artifacts/cf.npz           optional item-item collaborative weights from MovieLens - build_cf.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ARTIFACTS = Path(__file__).parent / "artifacts"
LIST_COLUMNS = ["genres", "keywords", "cast", "directors", "writers"]
DEFAULT_CF_WEIGHT = 0.85  # share of the collaborative signal, tuned with evaluate_cf.py


def _row_max_scale(x: np.ndarray) -> np.ndarray:
    top = x.max(axis=1, keepdims=True)
    return np.divide(x, top, out=np.zeros_like(x, dtype=np.float64), where=top > 0)


def blend(content: np.ndarray, cf: np.ndarray, cf_weight: float) -> np.ndarray:
    """Mix content similarity and collaborative scores, each scaled to [0, 1] by its row maximum.

    Rows with no collaborative signal (seeds nobody in MovieLens rated) fall back to content only.
    Zero the seed movies first so they don't set the scale.
    """
    content = _row_max_scale(np.atleast_2d(content))
    cf = _row_max_scale(np.clip(np.atleast_2d(cf), 0, None))
    weight = np.where(cf.max(axis=1, keepdims=True) > 0, cf_weight, 0.0)
    return (1 - weight) * content + weight * cf


class Recommender:
    def __init__(self, movies: pd.DataFrame, features: sp.csr_matrix, cf: sp.csr_matrix | None = None):
        self.movies = movies.reset_index(drop=True)
        self.features = features.tocsr()
        self.cf = cf.tocsr() if cf is not None else None
        self._row_of = {mid: i for i, mid in enumerate(self.movies["id"])}

    @classmethod
    def load(cls, path: Path = ARTIFACTS) -> "Recommender":
        movies = pd.read_parquet(path / "movies.parquet")
        for col in LIST_COLUMNS:
            movies[col] = movies[col].apply(list)
        cf = sp.load_npz(path / "cf.npz") if (path / "cf.npz").exists() else None
        return cls(movies, sp.load_npz(path / "features.npz"), cf)

    @property
    def has_cf(self) -> bool:
        return self.cf is not None

    def _rows(self, seed_ids: list[int]) -> list[int]:
        return [self._row_of[m] for m in seed_ids]

    def similarity(self, seed_ids: list[int]) -> np.ndarray:
        """Content cosine similarity of every movie to the (averaged) seed profile."""
        profile = np.asarray(self.features[self._rows(seed_ids)].sum(axis=0)).ravel()
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(len(self.movies))
        return self.features @ (profile / norm)

    def fan_scores(self, seed_ids: list[int]) -> np.ndarray:
        """Collaborative score: how strongly people who liked the seeds also liked each movie."""
        if self.cf is None:
            return np.zeros(len(self.movies))
        return np.asarray(self.cf[self._rows(seed_ids)].sum(axis=0), dtype=np.float64).ravel()

    def recommend(
        self,
        seed_ids: list[int],
        k: int = 10,
        quality_weight: float = 0.25,
        cf_weight: float = DEFAULT_CF_WEIGHT,
        genres: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
        min_rating: float = 0.0,
        min_votes: int = 0,
    ) -> pd.DataFrame:
        """Rank movies by a blend of content similarity and fan overlap, nudged towards well-rated titles.

        score = blend(content, collaborative, cf_weight) * ((1 - quality_weight) + quality_weight * quality)
        where ``quality`` is a 0-1 Bayesian (IMDb-style) weighted rating.
        """
        if not seed_ids:
            return self.movies.head(0).assign(similarity=[], fan_score=[], score=[])
        rows = self._rows(seed_ids)
        sim = self.similarity(seed_ids)
        fans = self.fan_scores(seed_ids)
        content = sim.copy()
        content[rows] = 0
        fans[rows] = 0
        hybrid = blend(content, fans, cf_weight)[0]
        m = self.movies
        score = hybrid * ((1 - quality_weight) + quality_weight * m["quality"].to_numpy())

        mask = hybrid > 0
        mask[rows] = False
        if genres:
            wanted = set(genres)
            mask &= m["genres"].apply(lambda g: bool(wanted.intersection(g))).to_numpy()
        if year_range:
            year = m["year"].fillna(0).to_numpy()
            mask &= (year >= year_range[0]) & (year <= year_range[1])
        mask &= m["vote_average"].to_numpy() >= min_rating
        mask &= m["vote_count"].to_numpy() >= min_votes

        idx = np.flatnonzero(mask)
        top = idx[np.argsort(-score[idx], kind="stable")[:k]]
        fan_share = _row_max_scale(np.clip(fans, 0, None)[None, :])[0]
        return m.iloc[top].assign(similarity=sim[top], fan_score=fan_share[top], score=score[top])

    def explain(self, seed_ids: list[int], movie: pd.Series, max_keywords: int = 4) -> dict[str, list[str]]:
        """What a recommended movie has in common with the seed movies."""
        rows = self._rows(seed_ids)
        seeds = self.movies.iloc[rows]

        def shared(col: str) -> list[str]:
            pool = {x for values in seeds[col] for x in values}
            return [x for x in movie[col] if x in pool]

        fans = []
        if self.cf is not None:
            j = self._row_of[int(movie["id"])]
            for r, title in zip(rows, seeds["title"]):
                neighbours = self.cf[r]
                if neighbours.nnz and neighbours[0, j] >= 0.25 * neighbours.data.max():
                    fans.append(title)

        return {
            "directors": shared("directors"),
            "cast": shared("cast"),
            "genres": shared("genres"),
            "keywords": shared("keywords")[:max_keywords],
            "fans": fans,
        }
