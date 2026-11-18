"""
train.py

Trains the models on the feature table made by features.py.

Data split (by year, so we never train on the future):
  - train on 2020-2023
  - validate on 2024 (used to pick the XGBoost settings)
  - test on 2025 (not touched here, only in evaluate.py)

Steps:
  1. Score two simple baselines on 2024: "higher-ranked player wins" and
     the Elo formula.
  2. Train logistic regression and XGBoost on 2020-2023 and score them on 2024.
     For XGBoost we try a few settings and keep the best one.
  3. Retrain both models on 2020-2024 with those settings and save them to models/.

How to run:  python src/train.py   (run features.py first)
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, FEATURES_FILE, PROJECT_DIR

MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALID_YEAR = 2024
TEST_YEAR = 2025

# The XGBoost settings we try. Small numbers on purpose: we only have a few
# thousand matches, so simple trees are less likely to overfit.
XGB_GRID = []
for depth in [2, 3, 4]:
    for rate in [0.03, 0.1]:
        for trees in [100, 300]:
            XGB_GRID.append({"max_depth": depth, "learning_rate": rate, "n_estimators": trees})


def load_features(path=FEATURES_FILE):
    """Read the feature table made by features.py."""
    return pd.read_csv(path, parse_dates=["date"])


def rank_prediction(df):
    """Baseline 1: the higher-ranked player wins.

    A lower rank number is better, so A is predicted to win when rank_diff < 0.
    This gives a 0 or 1 answer, not a probability.
    """
    return (df["rank_diff"] < 0).astype(int)


def elo_probability(df):
    """Baseline 2: the chance A wins from the normal Elo formula."""
    return 1 / (1 + 10 ** (-df["elo_diff"] / 400))


def bookmaker_probability(df):
    """The bookmakers' chance that A wins. Only used to compare, never as a feature.

    1/odds gives each player's implied chance, but the two add up to a bit
    more than 1 because the bookmaker takes a cut (the margin). Dividing by
    the total removes the margin so they add up to 1.

    Decimal odds must be bigger than 1. A few rows have bad odds (like -1 or
    0.967), so those rows are skipped.

    Returns the probabilities and a True/False list of which rows were usable.
    """
    usable = (df["odd_a"] > 1) & (df["odd_b"] > 1)
    implied_a = 1 / df.loc[usable, "odd_a"]
    implied_b = 1 / df.loc[usable, "odd_b"]
    return implied_a / (implied_a + implied_b), usable


def score(y_true, prob_a):
    """The four metrics we use, given the true labels and the chance A wins.

    - accuracy: how often the favourite (prob > 0.5) won
    - log loss: punishes confident wrong answers (lower is better)
    - Brier score: average squared error of the probability (lower is better)
    - ROC-AUC: how well it ranks winners above losers (higher is better)
    """
    return {
        "accuracy": accuracy_score(y_true, prob_a > 0.5),
        "log_loss": log_loss(y_true, prob_a),
        "brier": brier_score_loss(y_true, prob_a),
        "roc_auc": roc_auc_score(y_true, prob_a),
    }


def make_logistic_regression():
    """Logistic regression with scaling.

    The features have very different sizes (Elo diff is in the hundreds, form
    diff is between -1 and 1), so we scale them first. The pipeline makes sure
    the scaler is only fit on the training data.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def make_xgboost(settings):
    """XGBoost classifier with the given settings."""
    return XGBClassifier(
        max_depth=settings["max_depth"],
        learning_rate=settings["learning_rate"],
        n_estimators=settings["n_estimators"],
        subsample=0.8,         # each tree sees 80% of the rows, helps against overfitting
        colsample_bytree=0.8,  # and 80% of the features
        eval_metric="logloss",
        random_state=42,
    )


def tune_xgboost(X_train, y_train, X_valid, y_valid):
    """Try every setting in XGB_GRID and keep the one with the lowest
    log loss on the validation year."""
    best_settings = None
    best_loss = None
    for settings in XGB_GRID:
        model = make_xgboost(settings)
        model.fit(X_train, y_train)
        loss = log_loss(y_valid, model.predict_proba(X_valid)[:, 1])
        print(f"  {settings} -> validation log loss {loss:.4f}")
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_settings = settings
    return best_settings


def save_model(model, name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {path}")


def main():
    df = load_features()
    train = df[df["year"].isin(TRAIN_YEARS)]
    valid = df[df["year"] == VALID_YEAR]
    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_valid, y_valid = valid[FEATURE_COLUMNS], valid["target"]
    print(f"Train: {len(train)} matches (2020-2023), validation: {len(valid)} matches (2024)")

    results = []

    # Baselines. The rank baseline has no probability, so only accuracy makes sense.
    rank_acc = accuracy_score(y_valid, rank_prediction(valid))
    results.append({"method": "Higher rank wins", "accuracy": rank_acc,
                    "log_loss": np.nan, "brier": np.nan, "roc_auc": np.nan})
    results.append({"method": "Elo formula", **score(y_valid, elo_probability(valid))})

    # Logistic regression
    logreg = make_logistic_regression()
    logreg.fit(X_train, y_train)
    results.append({"method": "Logistic regression",
                    **score(y_valid, logreg.predict_proba(X_valid)[:, 1])})

    # XGBoost: pick the settings on 2024, then score the best one
    print("Tuning XGBoost on the 2024 validation year:")
    best = tune_xgboost(X_train, y_train, X_valid, y_valid)
    print(f"Best XGBoost settings: {best}")
    xgb = make_xgboost(best)
    xgb.fit(X_train, y_train)
    results.append({"method": "XGBoost", **score(y_valid, xgb.predict_proba(X_valid)[:, 1])})

    # Bookmakers, only for comparison
    book_prob, usable = bookmaker_probability(valid)
    results.append({"method": "Bookmakers", **score(y_valid[usable], book_prob)})
    print(f"Bookmakers: skipped {(~usable).sum()} validation rows with bad odds")

    results = pd.DataFrame(results)
    print("\nResults on the 2024 validation year:")
    print(results.round(4).to_string(index=False))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(os.path.join(RESULTS_DIR, "validation_metrics.csv"), index=False)

    # Retrain on 2020-2024 (more data) with the chosen settings, then save.
    # These final models are the ones tested on 2025 in evaluate.py.
    train_full = df[df["year"].isin(TRAIN_YEARS + [VALID_YEAR])]
    X_full, y_full = train_full[FEATURE_COLUMNS], train_full["target"]

    final_logreg = make_logistic_regression()
    final_logreg.fit(X_full, y_full)
    save_model(final_logreg, "logistic_regression.pkl")

    final_xgb = make_xgboost(best)
    final_xgb.fit(X_full, y_full)
    save_model(final_xgb, "xgboost.pkl")


if __name__ == "__main__":
    main()
