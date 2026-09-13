# 🎬 CineMatch — Movie Recommender

Pick one or more movies you like and get recommendations based on story, themes, cast and crew,
from the [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) dataset.

**Live app:** _coming soon_

## Features

- Search and pick up to 5 movies, and get recommendations that blend all of them
- Explanations on every card ("Same director Christopher Nolan · Starring Christian Bale · Themes: dc comics")
- Filters for genre, release year, minimum rating, and hiding obscure films
- A slider to favour highly rated films over pure similarity
- Shareable links: your selection is saved in the URL (`?movie=155&movie=27205`)
- Optional movie posters: add a free [TMDB API key](https://www.themoviedb.org/settings/api) as the `TMDB_API_KEY` secret

## How it works

1. **Data.** Movies and credits are joined on the TMDB id. The original notebook joined on title, which created duplicate rows for *Batman*, *The Host* and *Out of the Blue*.
2. **Features.** Each field is turned into its own TF-IDF vector:
   - overview, tagline and keywords, stemmed, with bigrams
   - keywords
   - genres
   - top 5 cast
   - directors
   - writers

   TF-IDF lets rare, informative terms such as a director or a niche theme count more than common ones like "Drama". The fields are then combined with tuned weights.
3. **Ranking.** Movies are ranked by cosine similarity to the selected movies. The score gets a small boost from an IMDb-style Bayesian weighted rating, so well-reviewed films rank higher than obscure ones with few votes.
4. **Size.** The app stores a 1.5 MB sparse feature matrix instead of the original 185 MB precomputed similarity matrix, so it deploys easily and starts fast.

## Evaluation

`python evaluate.py --tune` compares the original notebook model with the new one. It uses a check the models never see: **franchises identified from titles alone** (*Toy Story* / *Toy Story 2*, the *Harry Potter* films, and so on). For each of these films it counts how many of its sequels or siblings appear in the top 10. Weights were tuned on half of the franchises; the results below are for the other, held-out half.

| Model | Franchise recall@10 | Hit rate@10 | Avg. weighted rating of recs | Recs with < 50 votes |
|---|---|---|---|---|
| Original notebook (bag of words) | 0.595 | 0.669 | 6.60 | 33.1% |
| Improved (content only) | **0.831** | **0.850** | 6.59 | 25.2% |
| Improved + quality boost (default) | 0.828 | 0.850 | **6.64** | 25.6% |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the model or run the evaluation:

```bash
pip install -r requirements-dev.txt
python build_model.py      # writes artifacts/
python evaluate.py --tune  # grid-search weights, report held-out metrics
```

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `recommender.py` | Loads the artifacts; handles ranking, filters and explanations |
| `build_model.py` | Raw CSVs → `artifacts/movies.parquet` + `artifacts/features.npz` |
| `evaluate.py` | Offline comparison with the original model |
| `movie_recommender.ipynb` | Original exploratory notebook |

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), click **Create app**, choose the repo, branch `main`, and main file `app.py`.
3. Optional: under **Advanced settings → Secrets**, add `TMDB_API_KEY = "your-key"` to show posters.

---
Data from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.
