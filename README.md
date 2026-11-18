# tennis-ml-predictor

Predicts the winner of ATP men's singles matches (Grand Slams and Masters 1000,
2020-2025) and gives a win probability. This is a rebuild of the machine learning
part of my earlier [AI Tennis Match Predictor](https://github.com/Nasaa99/AI-Tennis-Match-Predictor).

The main goal of the rebuild is **no data leakage**: every feature for a match only
uses matches played before that day, and this is checked by tests.

## How to run

You need Python 3.9 or newer.

**Windows**
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

On a Mac, XGBoost also needs OpenMP: `brew install libomp`.

`run_all.py` runs everything from scratch in about a minute: features, leakage
tests, training and evaluation. You can also run each step on its own:

```bash
python src/features.py        # build features
python src/test_features.py   # leakage tests
python src/train.py           # train models
python src/evaluate.py        # test on 2025
```

## Dependencies

Listed with exact versions in `requirements.txt`: pandas, numpy, scikit-learn,
xgboost, shap and matplotlib.

## Where the code is

| File | What it does |
|---|---|
| `src/features.py` | Builds the features (Elo, surface Elo, rank, points, form, head-to-head, ...) in date order. Also has `make_features()` for predicting a new match. |
| `src/test_features.py` | Checks that future results and same-day results don't change a match's features. |
| `src/train.py` | **Training.** Baselines, logistic regression and XGBoost. Tunes XGBoost on 2024 and saves the final models to `models/`. |
| `src/evaluate.py` | Tests every method on 2025 and makes the results table and figures. |
| `run_all.py` | Runs all of the above in order. |

**Inference** (predicting a new match) uses `make_features()` from `src/features.py`
together with the saved model and player stats:

```python
import pickle
import pandas as pd
from features import FEATURE_COLUMNS, load_state, make_features

state = load_state()  # models/player_state.json: latest Elo, form, h2h of every player
model = pickle.load(open("models/logistic_regression.pkl", "rb"))

row = make_features(state, "Sinner J.", "Alcaraz C.", surface="Hard",
                    round_name="The Final", best_of=5)
X = pd.DataFrame([row])[FEATURE_COLUMNS]
prob = model.predict_proba(X)[0, 1]
print(f"Chance Sinner wins: {prob:.1%}")
```

This is the code the Flask API will use.

## Method

- **Data:** `Dataset/atp_matches_filtered.csv`, 5,908 matches from Jan 2020 to Sep 2025.
- **Features:** each match is one row, "player A vs player B", where A is picked
  at random (seed 42). Every feature is A minus B: Elo, surface Elo, rank, log rank
  ratio, ranking points, form over the last 5 and 10 matches, head-to-head,
  matches played, plus surface, round and best-of. The score, winner and betting
  odds are never features.
- **Elo:** starts at 1500, with K = 250 / (matches played + 5)^0.4, so new players'
  ratings move faster.
- **Split by year:** train 2020-2023, validate 2024 (to pick XGBoost settings),
  test 2025. The final models are retrained on 2020-2024.
- **Bookmakers** are only a benchmark: 1/odds, scaled so both players add up to 1.
  2 test matches with invalid odds are skipped.

## Results (2025 test year, 1,071 matches)

| Method | Accuracy | Log loss | Brier | ROC-AUC |
|---|---|---|---|---|
| Higher rank wins | 66.7% | - | - | - |
| Elo formula | 68.2% | 0.610 | 0.211 | 0.730 |
| Logistic regression | 67.1% | 0.592 | 0.204 | 0.744 |
| XGBoost | 67.1% | 0.596 | 0.206 | 0.741 |
| Bookmakers | 68.9% | 0.575 | 0.197 | 0.764 |

Logistic regression gives the best probabilities of our models. The bookmakers
are still better, which makes sense since they also know about injuries and news.
Figures are in `results/figures/`.
