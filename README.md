# 🎮 PlayCast — Predicting Game Owners on Steam

> **2nd-Year Linear Algebra Course Project**
> Applying **vector** and **matrix** theory together with Machine Learning
> to predict the estimated number of owners of a new game released on Steam.

---

## 1. What This Project Does

Game developers want to know roughly how many people will own/buy a game before it launches.
PlayCast takes a game's attributes (price, release year, tags, languages, publisher, categories, etc.)
and predicts the estimated number of owners, and it can also suggest **similar games** already on the market.

A GUI is provided so you can fill in the attributes and get a prediction instantly.

---

## 2. How It Works (Linear Algebra Connection)

The core idea is to turn each game into a **vector** and process them with **matrices**.

| Linear Algebra concept | Where it's used in the project |
|---|---|
| **One-hot vectors** | Text attributes (Tags / Languages / Publishers / Categories) are converted into binary vectors with `MultiLabelBinarizer` — each game becomes a vector in a 4,963-dimensional space |
| **Feature matrix X** | Stacking every game's vector forms a matrix of shape (number of games × 4,963) |
| **Normal Equation / Least Squares** | `LinearRegression` finds the coefficient vector **β** such that Xβ ≈ y by solving a linear system (β = (XᵀX)⁻¹Xᵀy) |
| **Dot product & Cosine Similarity** | Measures the angle between two games' feature vectors to find the most similar games |
| **Target log transform** | Predicts `log1p(owners)` and converts back with `expm1`, so the heavily skewed target fits a linear model better |

Data pipeline:

```
games.json ──▶ converted.csv ──▶ games_clean.csv ──▶ owners_model_mlr.joblib ──▶ GUI prediction
 (Kaggle)      tabular form        keep real games     trained model              + similar games
              convert_json_csv.py  clean_data.py       playcast.py                gui_play_cast.py
```

---

## 3. Project Structure

```
playcast/
├── README.md                    this document
├── CLAUDE.md                    technical overview (for reviewers / HR)
├── .gitignore
│
├── src/                         all source code
│   ├── convert_json_csv.py      1) convert games.json (Kaggle) → converted.csv
│   ├── clean_data.py            2) filter out non-games → games_clean.csv
│   ├── playcast.py              3) train the model + predict / find-similar functions
│   └── gui_play_cast.py         4) tkinter GUI to enter attributes and predict
│
├── data/                        data
│   ├── games_clean.csv          the actual training set (~10,400 games) — tracked in git
│   ├── games.json               Kaggle source (download yourself — not in git, 707MB)
│   └── converted.csv            intermediate file built by the pipeline (not in git, 505MB)
│
├── models/
│   └── owners_model_mlr.joblib  trained model (stores model + encoders + feature_columns)
│
└── reference/                   value lists found in the data (EDA output) for reference / report
    └── *_unique.csv / *_counts.csv   (tags, categories, genres, publishers, languages)
```

---

## 4. Installation & Usage

### Install dependencies
```bash
pip install numpy pandas scikit-learn joblib
```

### Use right away (a trained model is included)
The repo already ships `models/owners_model_mlr.joblib` and `data/games_clean.csv`, so you can open the GUI directly:
```bash
python src/gui_play_cast.py
```
Enter price / year / tags / languages / categories and click **🔮 Predict** to see the estimated owners,
or click **🧩 Similar Games** to see games that are most alike.

### Retrain the model from scratch (optional)
```bash
# 0) Download games.json from Kaggle (Steam Games Dataset) and place it at data/games.json

python src/convert_json_csv.py   # 1) games.json → data/converted.csv
python src/clean_data.py         # 2) converted.csv → data/games_clean.csv
python src/playcast.py           # 3) train → models/owners_model_mlr.joblib
python src/gui_play_cast.py      # 4) open the prediction GUI
```

> Every script resolves its paths relative to its own location (`__file__`), so you can run it from any directory.

---

## 5. Scope

**Supported:**
- Ingest raw Kaggle data (JSON) → clean it → train a Linear Regression model
- Predict the estimated owners of a game from its attributes (price, year, average playtime, reviews, tags, languages, publisher, categories)
- Recommend similar games via cosine similarity
- A GUI to search games, auto-fill attributes, and predict

**Limitations:**
- A linear (OLS) model meant to demonstrate Linear Algebra theory — not tuned for production-grade accuracy
- The model uses review counts (Positive/Negative) as features, which a truly new game does not have yet — you must supply estimates when predicting
- The data is a static Kaggle snapshot, not updated in real time

---

## 6. Data Source

**Steam Games Dataset (Kaggle)** — a JSON file keyed by each game's AppID.
After filtering with `clean_data.py` (removing non-game software/tools, price < 1 USD, no playtime, etc.),
about **10,400 games** remain for training.
