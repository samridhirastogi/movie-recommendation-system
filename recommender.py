"""Content-based movie recommender: loads precomputed artifacts and ranks movies.

Artifacts are produced by ``build_model.py``:
  artifacts/movies.parquet   one row per movie (metadata used for display/filters)
  artifacts/features.npz     L2-normalised sparse feature matrix (rows align with movies)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ARTIFACTS = Path(__file__).parent / "artifacts"
LIST_COLUMNS = ["genres", "keywords", "cast", "directors", "writers"]


class Recommender:
    def __init__(self, movies: pd.DataFrame, features: sp.csr_matrix):
        self.movies = movies.reset_index(drop=True)
        self.features = features.tocsr()
        self._row_of = {mid: i for i, mid in enumerate(self.movies["id"])}

    @classmethod
    def load(cls, path: Path = ARTIFACTS) -> "Recommender":
        movies = pd.read_parquet(path / "movies.parquet")
        for col in LIST_COLUMNS:
            movies[col] = movies[col].apply(list)
        return cls(movies, sp.load_npz(path / "features.npz"))

    def similarity(self, seed_ids: list[int]) -> np.ndarray:
        """Cosine similarity of every movie to the (averaged) seed profile."""
        rows = [self._row_of[m] for m in seed_ids]
        profile = np.asarray(self.features[rows].sum(axis=0)).ravel()
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(len(self.movies))
        return self.features @ (profile / norm)

    def recommend(
        self,
        seed_ids: list[int],
        k: int = 10,
        quality_weight: float = 0.25,
        genres: list[str] | None = None,
        year_range: tuple[int, int] | None = None,
        min_rating: float = 0.0,
        min_votes: int = 0,
    ) -> pd.DataFrame:
        """Rank movies by content similarity, nudged towards well-rated titles.

        score = similarity * ((1 - quality_weight) + quality_weight * quality)
        where ``quality`` is a 0-1 Bayesian (IMDb-style) weighted rating.
        """
        if not seed_ids:
            return self.movies.head(0).assign(similarity=[], score=[])
        sim = self.similarity(seed_ids)
        m = self.movies
        score = sim * ((1 - quality_weight) + quality_weight * m["quality"].to_numpy())

        mask = ~m["id"].isin(seed_ids).to_numpy() & (sim > 0)
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
        return m.iloc[top].assign(similarity=sim[top], score=score[top])

    def explain(self, seed_ids: list[int], movie: pd.Series, max_keywords: int = 4) -> dict[str, list[str]]:
        """What a recommended movie has in common with the seed movies."""
        seeds = self.movies.iloc[[self._row_of[m] for m in seed_ids]]

        def shared(col: str) -> list[str]:
            pool = {x for values in seeds[col] for x in values}
            return [x for x in movie[col] if x in pool]

        return {
            "directors": shared("directors"),
            "cast": shared("cast"),
            "genres": shared("genres"),
            "keywords": shared("keywords")[:max_keywords],
        }
