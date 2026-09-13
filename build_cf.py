"""Build the collaborative-filtering artifact from MovieLens ratings.

    python build_cf.py path/to/ml-25m.zip

Learns "people who liked X also liked Y" for the TMDB 5000 catalogue from
MovieLens 25M, using item-item cosine similarity with shrinkage on "liked"
(4+ star) ratings. In evaluate_cf.py it beat EASE (Steck, 2019), which is also
implemented here for comparison.

Only derived item-to-item weights are written (artifacts/cf.npz). Raw ratings
are never saved, because the MovieLens licence forbids redistributing them.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import scipy.sparse as sp

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
LIKE_THRESHOLD = 4.0   # a rating of 4+ stars counts as "liked"
DEFAULT_SHRINK = 500.0  # cosine shrinkage, tuned with evaluate_cf.py
TOP_K = 100            # neighbours kept per movie


def _read_csv(zf: zipfile.ZipFile, name: str, types: dict[str, pa.DataType]) -> pa.Table:
    member = next(n for n in zf.namelist() if n.split("/")[-1] == name)
    with zf.open(member) as f:
        options = pacsv.ConvertOptions(include_columns=list(types), column_types=types)
        return pacsv.read_csv(f, convert_options=options)


def load_likes(ml_zip: Path, tmdb_ids) -> sp.csr_matrix:
    """Binary users x movies matrix of ratings >= LIKE_THRESHOLD; columns follow ``tmdb_ids``."""
    col_of = {int(t): j for j, t in enumerate(tmdb_ids)}
    with zipfile.ZipFile(ml_zip) as zf:
        links = _read_csv(zf, "links.csv", {"movieId": pa.int32(), "tmdbId": pa.int64()}).to_pandas()
        ratings = _read_csv(zf, "ratings.csv", {"userId": pa.int32(), "movieId": pa.int32(), "rating": pa.float32()})
    liked = ratings.filter(pc.greater_equal(ratings["rating"], LIKE_THRESHOLD))
    del ratings
    users = liked["userId"].to_numpy()
    movie_ids = liked["movieId"].to_numpy()
    del liked

    links = links.dropna()
    links = links[links["tmdbId"].astype(int).isin(col_of)]
    lookup = np.full(int(movie_ids.max()) + 1, -1, dtype=np.int32)
    in_range = links["movieId"] < len(lookup)
    lookup[links.loc[in_range, "movieId"].to_numpy()] = links.loc[in_range, "tmdbId"].astype(int).map(col_of).to_numpy()

    cols = lookup[movie_ids]
    keep = cols >= 0
    _, rows = np.unique(users[keep], return_inverse=True)
    x = sp.csr_matrix((np.ones(keep.sum(), dtype=np.float32), (rows, cols[keep])),
                      shape=(rows.max() + 1, len(col_of)))
    x.data[:] = 1.0  # duplicate MovieLens ids for one TMDB movie were summed
    return x


def cooccurrence(x: sp.csr_matrix) -> np.ndarray:
    """Dense movie x movie matrix: how many users liked both (the diagonal is each movie's like count)."""
    return (x.T @ x).toarray().astype(np.float32)


def item_cosine(g: np.ndarray, shrink: float = DEFAULT_SHRINK) -> np.ndarray:
    """Item-item cosine similarity; shrinkage damps pairs of rarely liked movies."""
    n = np.sqrt(np.diag(g))
    s = g / (np.outer(n, n) + shrink)
    np.fill_diagonal(s, 0.0)
    return s.astype(np.float32)


def ease(g: np.ndarray, lam: float) -> np.ndarray:
    """EASE item-item weights B with zero diagonal: score(user) = x_user @ B."""
    a = g.astype(np.float64)
    a[np.diag_indices_from(a)] += lam
    p = np.linalg.inv(a)
    b = p / -np.diag(p)
    np.fill_diagonal(b, 0.0)
    return b.astype(np.float32)


def top_k(weights: np.ndarray, k: int = TOP_K) -> sp.csr_matrix:
    """Keep each movie's k strongest positive neighbours as a sparse matrix."""
    idx = np.argpartition(-weights, k, axis=1)[:, :k]
    vals = np.take_along_axis(weights, idx, axis=1)
    rows = np.repeat(np.arange(weights.shape[0]), k)
    m = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=weights.shape)
    m.data[m.data < 0] = 0
    m.eliminate_zeros()
    return m


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python build_cf.py path/to/ml-25m.zip")
    movies = pd.read_parquet(ARTIFACTS / "movies.parquet", columns=["id"])
    x = load_likes(Path(sys.argv[1]), movies["id"])
    users, n_likes = x.shape[0], x.nnz
    g = cooccurrence(x)
    del x
    likes = np.diag(g)
    cf = top_k(item_cosine(g, DEFAULT_SHRINK))
    sp.save_npz(ARTIFACTS / "cf.npz", cf)
    print(f"{users:,} users, {n_likes:,} likes; {(likes >= 20).sum():,}/{len(likes):,} movies liked by 20+ users; "
          f"{cf.nnz:,} neighbour weights -> {ARTIFACTS / 'cf.npz'}")


if __name__ == "__main__":
    main()
