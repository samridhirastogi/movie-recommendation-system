"""Evaluate content, collaborative and hybrid recommenders on real MovieLens users.

    python evaluate_cf.py path/to/ml-25m.zip

Protocol (mirrors how the app is used):
  * 10% of MovieLens users are held out; collaborative models are fitted on the rest only.
  * For each held-out user who liked >= 8 catalogue movies, 3 of their liked movies are the
    seeds, and the model must retrieve the user's other liked movies in its top 10.
  * Half of those users (and half of the title-derived franchises) choose the model and its
    settings; the other half are only used for the final report.
  * The default blend weight is the smallest cf_weight within TOLERANCE of the best validation
    NDCG: it keeps as much story/cast matching (franchises, variety) as accuracy allows.

Every model is scored through Recommender.recommend, the exact code path the app uses.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from build_cf import cooccurrence, ease, item_cosine, load_likes, top_k
from evaluate import franchise_groups
from recommender import Recommender

K = 10
N_SEEDS = 3
MIN_LIKES = 8
MAX_CASES = 4000
SHRINKS = (5.0, 20.0, 100.0, 500.0)
LAMBDAS = (5000.0, 20000.0)
CF_WEIGHTS = (0.5, 0.7, 0.85, 0.95, 1.0)
TOLERANCE = 0.02
rng = np.random.default_rng(42)


def make_cases(x_test) -> list[tuple[np.ndarray, np.ndarray]]:
    cases = []
    for u in rng.permutation(x_test.shape[0]):
        liked = x_test.indices[x_test.indptr[u]:x_test.indptr[u + 1]]
        if len(liked) >= MIN_LIKES:
            seeds = rng.choice(liked, N_SEEDS, replace=False)
            cases.append((seeds, np.setdiff1d(liked, seeds)))
            if len(cases) == MAX_CASES:
                break
    return cases


def run(recommend_rows, cases, movies: pd.DataFrame) -> dict[str, float]:
    discount = 1 / np.log2(np.arange(2, K + 2))
    recall, ndcg, ratings, shown = [], [], [], set()
    for seeds, relevant in cases:
        top = recommend_rows(seeds)
        hit = np.isin(top, relevant)
        n = min(K, len(relevant))
        recall.append(hit.sum() / n)
        ndcg.append((hit * discount[:len(top)]).sum() / discount[:n].sum())
        ratings.extend(movies["weighted_rating"].to_numpy()[top])
        shown.update(top.tolist())
    return {"recall@10": np.mean(recall), "ndcg@10": np.mean(ndcg),
            "avg_rating": np.mean(ratings), "coverage": len(shown) / len(movies)}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python evaluate_cf.py path/to/ml-25m.zip")
    content = Recommender.load()
    movies, features = content.movies, content.features
    ids = movies["id"].to_numpy()

    x = load_likes(Path(sys.argv[1]), ids)
    held_out = rng.random(x.shape[0]) < 0.1
    x_train = x[~held_out]
    cases = make_cases(x[held_out])
    val, test = cases[::2], cases[1::2]
    print(f"{x.shape[0]:,} users ({held_out.sum():,} held out), {x.nnz:,} likes; "
          f"{len(val)} validation / {len(test)} test users", flush=True)
    del x
    g = cooccurrence(x_train)
    del x_train
    popular = np.argsort(-np.diag(g))[:K + N_SEEDS]

    groups = franchise_groups(movies)
    half = lambda h: [(np.array([i]), grp[grp != i]) for grp in groups
                      if zlib.crc32(movies["title"][grp[0]].encode()) % 2 == h for i in grp]
    franchise_val, franchise_test = half(0), half(1)

    def via(rec: Recommender, cf_weight: float):
        return lambda seeds: rec.recommend([int(i) for i in ids[seeds]], k=K, cf_weight=cf_weight).index.to_numpy()

    candidates = {f"item cosine shrink={s:g}": top_k(item_cosine(g, s)) for s in SHRINKS}
    candidates.update({f"EASE lambda={lam:g}": top_k(ease(g, lam)) for lam in LAMBDAS})
    del g

    val_ndcg = {}
    for name, cf in candidates.items():
        rec = Recommender(movies, features, cf)
        for w in CF_WEIGHTS:
            r = run(via(rec, w), val, movies)
            fr = run(via(rec, w), franchise_val, movies)["recall@10"]
            val_ndcg[name, w] = r["ndcg@10"]
            print(f"  val  {name:24s} cf_weight={w:<4} ndcg@10={r['ndcg@10']:.4f} "
                  f"recall@10={r['recall@10']:.4f} franchise={fr:.3f} coverage={r['coverage']:.3f}", flush=True)

    best_name = max(candidates, key=lambda n: max(val_ndcg[n, w] for w in CF_WEIGHTS))
    best = max(val_ndcg[best_name, w] for w in CF_WEIGHTS)
    default_w = min(w for w in CF_WEIGHTS if val_ndcg[best_name, w] >= (1 - TOLERANCE) * best)
    print(f"best model: {best_name}; default cf_weight={default_w} "
          f"(smallest weight within {TOLERANCE:.0%} of best validation NDCG {best:.4f})", flush=True)

    rec = Recommender(movies, features, candidates[best_name])
    models = {"popularity (most liked)": lambda seeds: popular[~np.isin(popular, seeds)][:K],
              "content only (cf_weight=0)": via(content, 0.0)}
    for w in CF_WEIGHTS:
        models[f"hybrid cf_weight={w}" + (" <- default" if w == default_w else "")] = via(rec, w)
    rows = {}
    for label, fn in models.items():
        rows[label] = run(fn, test, movies)
        rows[label]["franchise_recall@10"] = run(fn, franchise_test, movies)["recall@10"]
    print(f"\nTEST ({len(test)} held-out users; {best_name})")
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
