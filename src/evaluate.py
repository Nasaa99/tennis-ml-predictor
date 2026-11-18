"""
evaluate.py

Tests everything on 2025, the year no model has seen.

Compares five methods: the two baselines (higher rank wins, Elo formula),
logistic regression, XGBoost, and the bookmakers.

Saves:
  - results/metrics.csv
  - results/figures/comparison.png     accuracy and log loss for every method
  - results/figures/calibration.png    are the predicted chances honest?
  - results/figures/feature_importance.png
  - results/figures/shap_summary.png

How to run:  python src/evaluate.py   (run features.py and train.py first)
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

from features import FEATURE_COLUMNS
from train import (MODELS_DIR, RESULTS_DIR, TEST_YEAR, bookmaker_probability,
                   elo_probability, load_features, rank_prediction, score)

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Colours: grey for the simple baselines, blue for our models, orange for the bookmakers
GREY = "#9a9890"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
METHOD_COLOURS = {
    "Higher rank wins": GREY,
    "Elo formula": GREY,
    "Logistic regression": BLUE,
    "XGBoost": BLUE,
    "Bookmakers": ORANGE,
}


def load_model(name):
    with open(os.path.join(MODELS_DIR, name), "rb") as f:
        return pickle.load(f)


def compute_metrics(test, logreg, xgb):
    """Score every method on the test year. Returns the table and the
    probabilities (we need those again for the calibration plot)."""
    y = test["target"]
    X = test[FEATURE_COLUMNS]

    probs = {
        "Elo formula": elo_probability(test),
        "Logistic regression": pd.Series(logreg.predict_proba(X)[:, 1], index=test.index),
        "XGBoost": pd.Series(xgb.predict_proba(X)[:, 1], index=test.index),
    }
    book_prob, usable = bookmaker_probability(test)
    probs["Bookmakers"] = book_prob

    rows = []
    # The rank baseline only says who wins (no probability), so only accuracy applies
    rows.append({"method": "Higher rank wins", "matches": len(test),
                 "accuracy": accuracy_score(y, rank_prediction(test)),
                 "log_loss": np.nan, "brier": np.nan, "roc_auc": np.nan})
    for method, prob in probs.items():
        rows.append({"method": method, "matches": len(prob), **score(y[prob.index], prob)})

    skipped = (~usable).sum()
    return pd.DataFrame(rows), probs, skipped


def plot_comparison(metrics):
    """Two bar charts side by side: accuracy (higher is better) and
    log loss (lower is better). Two separate charts because the numbers
    are on different scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    colours = [METHOD_COLOURS[m] for m in metrics["method"]]
    ax1.barh(metrics["method"], metrics["accuracy"] * 100, color=colours, height=0.6)
    ax1.set_title("Accuracy (%), higher is better", loc="left")
    ax1.set_xlim(50, 75)
    for i, value in enumerate(metrics["accuracy"] * 100):
        ax1.text(value + 0.3, i, f"{value:.1f}", va="center", fontsize=9)

    # The rank baseline has no log loss, so leave it out of the second chart
    has_loss = metrics.dropna(subset=["log_loss"])
    colours = [METHOD_COLOURS[m] for m in has_loss["method"]]
    ax2.barh(has_loss["method"], has_loss["log_loss"], color=colours, height=0.6)
    ax2.set_title("Log loss, lower is better", loc="left")
    ax2.set_xlim(0.5, 0.65)
    for i, value in enumerate(has_loss["log_loss"]):
        ax2.text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=9)

    for ax in (ax1, ax2):
        ax.invert_yaxis()  # first method at the top
        ax.grid(axis="x", color="#e5e4df")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(f"All methods on the {TEST_YEAR} test year", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "comparison.png"), dpi=150)
    plt.close(fig)


def plot_calibration(test, probs):
    """Calibration curve: group matches by predicted chance (e.g. all the
    "about 70%" predictions) and check how often A really won. A perfect
    model sits on the diagonal line."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1, label="Perfect")

    line_colours = {"Elo formula": "#1baf7a", "Logistic regression": BLUE,
                    "XGBoost": "#6250d6", "Bookmakers": ORANGE}
    for method, prob in probs.items():
        y = test.loc[prob.index, "target"]
        # 10 bins with the same number of matches in each
        actual, predicted = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax.plot(predicted, actual, marker="o", markersize=5, linewidth=2,
                color=line_colours[method], label=method)

    ax.set_xlabel("Predicted chance that A wins")
    ax.set_ylabel("How often A actually won")
    ax.set_title(f"Calibration on {TEST_YEAR}", loc="left", fontweight="bold")
    ax.grid(color="#e5e4df")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "calibration.png"), dpi=150)
    plt.close(fig)


def plot_feature_importance(xgb):
    """How much each feature helped XGBoost's trees (gain), largest at the top."""
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.index, importance.values, color=BLUE, height=0.6)
    ax.set_title("XGBoost feature importance (gain)", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e4df")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_shap(xgb, test):
    """SHAP summary: each dot is one match. It shows how much each feature
    pushed that prediction towards A winning (right) or losing (left), and
    the colour shows if the feature value was high or low."""
    X = test[FEATURE_COLUMNS]
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP summary for XGBoost ({TEST_YEAR})", loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"), dpi=150)
    plt.close()


def print_summary(metrics, skipped):
    """A short summary of the results in plain English."""
    m = metrics.set_index("method")
    best_model = m.loc[["Logistic regression", "XGBoost"], "log_loss"].idxmin()
    print("\nSummary:")
    print(f"- Picking the higher-ranked player is right {m.loc['Higher rank wins', 'accuracy']:.1%} of the time.")
    print(f"- The Elo formula alone gets {m.loc['Elo formula', 'accuracy']:.1%} "
          f"(log loss {m.loc['Elo formula', 'log_loss']:.3f}).")
    print(f"- Logistic regression gets {m.loc['Logistic regression', 'accuracy']:.1%} "
          f"(log loss {m.loc['Logistic regression', 'log_loss']:.3f}), "
          f"XGBoost gets {m.loc['XGBoost', 'accuracy']:.1%} "
          f"(log loss {m.loc['XGBoost', 'log_loss']:.3f}).")
    print(f"- The bookmakers get {m.loc['Bookmakers', 'accuracy']:.1%} "
          f"(log loss {m.loc['Bookmakers', 'log_loss']:.3f}); "
          f"{skipped} matches were skipped because of bad odds.")
    gap = m.loc[best_model, "log_loss"] - m.loc["Bookmakers", "log_loss"]
    print(f"- Our best model by log loss is {best_model}. It is {gap:.3f} behind the "
          f"bookmakers, who also know things our data doesn't (injuries, fitness, news).")


def main():
    df = load_features()
    test = df[df["year"] == TEST_YEAR]
    logreg = load_model("logistic_regression.pkl")
    xgb = load_model("xgboost.pkl")
    print(f"Test year {TEST_YEAR}: {len(test)} matches")

    metrics, probs, skipped = compute_metrics(test, logreg, xgb)
    print(metrics.round(4).to_string(index=False))

    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    plot_comparison(metrics)
    plot_calibration(test, probs)
    plot_feature_importance(xgb)
    plot_shap(xgb, test)
    print(f"Saved metrics.csv and 4 figures to {RESULTS_DIR}")

    print_summary(metrics, skipped)


if __name__ == "__main__":
    main()
