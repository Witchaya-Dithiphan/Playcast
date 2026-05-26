# CLAUDE.md — Technical Project Overview

This document summarizes the project for code reviewers / HR / anyone reading it for the first time.
(For usage instructions, see [README.md](README.md).)

---

## What the Project Is

**PlayCast** — a **2nd-year Linear Algebra course project**.
Goal: apply **vector and matrix** theory in practice through Machine Learning
to **predict the estimated number of owners of a game on Steam**.

Written in **Python**, using `pandas` (data handling), `scikit-learn` (model / vectors),
`numpy` (numerical computation), and `tkinter` (GUI).

---

## Why It Relates to Linear Algebra

The project shows how concepts from a Linear Algebra course solve a real problem:

1. **Building vectors from text data** — each game has text attributes (tags, languages, categories).
   We encode them as one-hot vectors with `MultiLabelBinarizer`, so one game becomes a vector in a 4,963-dimensional space.

2. **Matrices and linear regression** — the vectors of all games are stacked into a matrix X,
   then Linear Regression finds the coefficient vector β such that Xβ ≈ y
   (Least Squares / Normal Equation: β = (XᵀX)⁻¹Xᵀy).

3. **Similarity via cosine similarity** — measures the angle between two games' feature vectors
   to find the "most similar games" (using dot product / vector norms).

---

## Architecture / Data Pipeline

```
games.json ──▶ converted.csv ──▶ games_clean.csv ──▶ owners_model_mlr.joblib ──▶ GUI
(Kaggle)       tabular form        keep real games     trained model              predict + suggest
```

| File (`src/`) | Role |
|---|---|
| `convert_json_csv.py` | Read the Kaggle JSON, convert dict/list fields to strings, save as CSV |
| `clean_data.py` | Filter out non-games (software/tools), price < 1 USD, no playtime, no English support, etc. |
| `playcast.py` | Build features (vector encoding), train Linear Regression, plus `predict_new_game` and `find_similar_games` |
| `gui_play_cast.py` | tkinter GUI: search games, auto-fill attributes, predict / find-similar buttons |

**The saved model** (`models/owners_model_mlr.joblib`) is a dict with three parts:
`model` (LinearRegression), `encoders` (four MultiLabelBinarizers), and `feature_columns` (the ordered list of 4,963 columns).

---

## Features Used for Training

- **Numeric:** price, release year, average playtime, log of positive/negative reviews, positive-review ratio
- **One-hot vectors:** Tags (442), Supported languages (57), Publishers (4,418), Categories (40)
- **Target (y):** `log1p(estimated owners)` — predicted, then converted back with `expm1`

Cleaned dataset: **~10,400 games**

---

## Quick Start

```bash
pip install numpy pandas scikit-learn joblib
python src/gui_play_cast.py        # works with the included trained model
```
Retrain: `convert_json_csv.py → clean_data.py → playcast.py` (requires `data/games.json` from Kaggle first).

---

## Notes for Reviewers

- Every script resolves paths via `__file__`, so it runs from any working directory.
- Large data files (`games.json` 707MB, `converted.csv` 505MB) are not in git
  (download/regenerate them via the pipeline) — only `games_clean.csv`, the actual training set, is in the repo.
- This is an educational project, focused on demonstrating applied Linear Algebra rather than production-grade accuracy.
