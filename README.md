# 🎬 CineMatch — Hybrid Movie Recommender

Pick movies you like and get recommendations that combine two things:

- **What the films are about:** story, themes, cast and crew, from the [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) dataset.
- **What people who liked them also enjoyed:** 25 million ratings from [MovieLens](https://grouplens.org/datasets/movielens/).

**Live app:** _coming soon_

## Features

- Pick up to 5 movies and get recommendations that blend all of them
- Explanations on every card, e.g. *"Fans of Inception also liked this · Same director Christopher Nolan · Themes: dream, heist"*
- A **Story & cast ↔ What fans liked** slider to control the mix
- Filters for genre, release year, minimum rating, and hiding obscure films
- A slider to favour highly rated films
- Shareable links: your selection is saved in the URL (`?movie=155&movie=27205`)
- Optional movie posters: add a free [TMDB API key](https://www.themoviedb.org/settings/api) as the `TMDB_API_KEY` secret

## How it works

### 1. Content model: what a film is about (`build_model.py`)
- **Data.** TMDB movies and credits are joined on the TMDB id. The original notebook joined on title, which duplicated rows for films that share a name.
- **Features.** Each field becomes its own TF-IDF vector, so rare, informative terms (a director, a niche theme) outweigh common ones like "Drama". The weights were tuned with `evaluate.py`:

  | Field | Weight |
  |---|---|
  | Overview, tagline and keywords (stemmed, with bigrams) | 1.0 |
  | Keywords | 0.5 |
  | Writers | 0.5 |
  | Top 5 cast | 0.4 |
  | Directors | 0.4 |
  | Genres | 0.3 |

- **Similarity.** Cosine similarity between the selected movies and every other movie.

### 2. Collaborative model: what fans liked (`build_cf.py`)
- **Likes.** A MovieLens rating of 4 stars or more counts as a "like". That gives **8.78 million likes from 162,205 users** on the 4,803 catalogue movies, matched through MovieLens `links.csv`.
- **Similarity.** Item-to-item cosine similarity with shrinkage:
  `sim(i, j) = users who liked both / (√(likes of i × likes of j) + 500)`.
  The shrinkage term stops pairs of rarely liked movies from looking falsely similar.
- **Stored model.** Each movie keeps its 100 strongest neighbours.
- **Alternative tried.** EASE (Steck, 2019) scored lower in validation: NDCG@10 0.421 vs 0.467.

### 3. Hybrid ranking (`recommender.py`)
```
score = [(1 − w) · content + w · fans] × (0.75 + 0.25 · quality)
```
- `content` and `fans` are each scaled to 0–1 for the current query.
- `w` is the blend weight: **0.85** by default, adjustable with the slider.
- `quality` is an IMDb-style Bayesian weighted rating.
- Movies with no MovieLens data fall back to the content model.

## Evaluation

### A. Predicting real users' taste (`evaluate_cf.py`, MovieLens 25M)

**Setup.** 10% of MovieLens users were held out, and the collaborative model never saw them during training. For each held-out user who liked at least 8 catalogue movies:
- the model is given 3 of their liked movies (like picking movies in the app);
- it must find the user's *other* liked movies in its top 10.

**Tuning and reporting.** One group of 2,000 users was used to choose the model and settings. The results below are for a separate group of **2,000 test users**.

| Model | Recall@10 | NDCG@10 | Avg. rating of recs | Catalogue coverage | Franchise recall@10 |
|---|---|---|---|---|---|
| Popularity (most liked) | 0.344 | 0.363 | 8.15 | 0.3% | 0.008 |
| Content only (previous version) | 0.114 | 0.136 | 6.86 | 57.9% | **0.828** |
| Hybrid, w = 0.5 | 0.392 | 0.404 | 7.64 | 17.7% | 0.778 |
| Hybrid, w = 0.7 | 0.434 | 0.457 | 7.75 | 13.1% | 0.722 |
| **Hybrid, w = 0.85 (default)** | **0.442** | **0.469** | 7.78 | 11.5% | 0.678 |
| Collaborative only, w = 1.0 | 0.442 | 0.469 | 7.80 | 10.5% | 0.641 |

- **Accuracy.** The hybrid predicts what users like **3.4× better** than content-only matching (NDCG), and **29% better** than recommending the most popular films. Popularity is a strong baseline here, because most MovieLens users rate popular films.
- **Why 0.85.** It's the smallest blend weight within 2% of the best validation score. On test users it matches collaborative-only accuracy while keeping more story/cast matching (franchise recall 0.678 vs 0.641) and more variety.
- **Trade-off.** Lowering `w` recovers more same-franchise and less mainstream picks, at some cost in accuracy. The slider in the app lets users make that trade themselves.

### B. Franchise recall of the content model (`evaluate.py`)

This test uses franchises identified from titles alone (*Toy Story* / *Toy Story 2*, the *Harry Potter* films, and so on); the models never see titles. It measures how many of a film's siblings appear in its top 10, on held-out franchises:

| Model | Franchise recall@10 | Hit rate@10 | Avg. weighted rating | Recs with < 50 votes |
|---|---|---|---|---|
| Original notebook (bag of words) | 0.595 | 0.669 | 6.60 | 33.1% |
| Improved content model | **0.831** | **0.850** | 6.59 | 25.2% |
| Improved + quality boost | 0.828 | 0.850 | **6.64** | 25.6% |

## Efficiency

| | Original notebook | This version |
|---|---|---|
| Model files | 185 MB dense similarity matrix | **5.1 MB** total: content features 1.4 MB, movie data 1.6 MB, collaborative neighbours 2.1 MB |
| Time per recommendation | n/a | **2.3 ms** median, 3.5 ms p95 (hybrid, 3 selected movies) |
| Model load time | n/a | 0.7 s |

Similarity is computed when you ask for recommendations, not stored in advance, using sparse matrices.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the models or rerun the evaluations:

```bash
pip install -r requirements-dev.txt
python build_model.py                    # content model  -> artifacts/movies.parquet, features.npz
python evaluate.py --tune                # franchise evaluation of the content model
# Download ml-25m.zip from https://grouplens.org/datasets/movielens/25m/ (do not commit it)
python build_cf.py path/to/ml-25m.zip    # collaborative model -> artifacts/cf.npz
python evaluate_cf.py path/to/ml-25m.zip # user-based evaluation and tuning
```

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `recommender.py` | Hybrid ranking, filters and explanations |
| `build_model.py` | TMDB CSVs → content features |
| `build_cf.py` | MovieLens ratings → item-to-item neighbours |
| `evaluate.py` | Franchise evaluation (content model vs original notebook) |
| `evaluate_cf.py` | Held-out user evaluation (content vs collaborative vs hybrid) |
| `movie_recommender.ipynb` | Original exploratory notebook |

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose the repo, branch `main`, and main file `app.py`.
3. Optional: under **Advanced settings → Secrets**, add `TMDB_API_KEY = "your-key"` to show posters.

## Data and licences

- **TMDB 5000 Movie Dataset.** This product uses the TMDB API but is not endorsed or certified by TMDB.
- **MovieLens 25M** by GroupLens Research, University of Minnesota. It is used here only in this non-commercial project, and nothing here implies endorsement by the University of Minnesota or GroupLens. As the MovieLens licence requires, **the ratings are not redistributed**: this repository contains only derived movie-to-movie similarity scores (`artifacts/cf.npz`). Citation:

  > F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
