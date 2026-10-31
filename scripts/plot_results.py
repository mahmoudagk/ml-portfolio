"""Create reproducible diagnostic images from the saved models and cleaned data."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
NAVY = "#17324D"
TEAL = "#159A9C"
GOLD = "#E6A23C"
MUTED = "#66788A"
GRID = "#DDE6EE"
PANEL = "#FFFFFF"
CANVAS = "#F4F7FB"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def canvas(title: str, subtitle: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=False)
    fig.patch.set_facecolor(CANVAS)
    fig.text(0.055, 0.94, title, fontsize=23, fontweight="bold", color=NAVY, va="top")
    fig.text(0.057, 0.885, subtitle, fontsize=11.5, color=MUTED, va="top")
    fig.text(0.057, 0.045, "MODEL EVALUATION  •  HELD-OUT DATA", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.79, bottom=0.14, wspace=0.28)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
    return fig, axes


def plot_roc(ax, y_true: pd.Series, probability: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(y_true, probability)
    score = roc_auc_score(y_true, probability)
    ax.plot(false_positive, true_positive, color=TEAL, linewidth=3, label=f"ROC-AUC = {score:.3f}")
    ax.fill_between(false_positive, true_positive, color=TEAL, alpha=0.12)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.4, label="Random baseline")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("Ranking performance", loc="left", fontsize=14, pad=15)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    return score


def plot_confusion(ax, y_true: pd.Series, predicted: np.ndarray, threshold: float = 0.5) -> None:
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Predictions at threshold {threshold:.2f}", loc="left", fontsize=14, pad=15)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["No", "Yes"])
    ax.set_yticks([0, 1], labels=["No", "Yes"])
    midpoint = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > midpoint else NAVY
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=16, fontweight="bold", color=color)


def bank_marketing() -> None:
    project = ROOT / "projects" / "bank-marketing"
    data = pd.read_csv(project / "data" / "processed" / "bank_marketing_clean.csv")
    features = [column for column in data.columns if column not in {"y", "duration"}]
    X_train, X_test, _, y_test = train_test_split(
        data[features], data["y"].map({"no": 0, "yes": 1}),
        test_size=0.2, random_state=42, stratify=data["y"],
    )
    saved = joblib.load(project / "artifacts" / "bank_marketing_model.joblib")
    if isinstance(saved, dict):
        model = saved["model"]
        threshold = float(saved["threshold"])
    else:
        model = saved
        threshold = 0.5
    probability = model.predict_proba(X_test)[:, 1]
    predicted = (probability >= threshold).astype(int)
    fig, axes = canvas(
        "Bank marketing | Term-deposit response",
        f"Stratified 60/20/20 split  •  {len(X_test):,} test customers  •  threshold {threshold:.2f} selected on validation",
    )
    plot_roc(axes[0], y_test, probability)
    plot_confusion(axes[1], y_test, predicted, threshold)
    fig.savefig(project / "images" / "results.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def adult_income() -> None:
    project = ROOT / "projects" / "adult-income"
    test = pd.read_csv(project / "data" / "processed" / "adult_test_clean.csv")
    features = [column for column in test.columns if column != "income"]
    X_test = test[features]
    y_test = test["income"].map({"<=50K": 0, ">50K": 1})
    model = joblib.load(project / "artifacts" / "adult_income_model.joblib")
    probability = model.predict_proba(X_test)[:, 1]
    predicted = model.predict(X_test)
    fig, axes = canvas(
        "Adult census | Income classification",
        "Official UCI test split  •  16,281 records  •  Historical data; educational use only",
    )
    plot_roc(axes[0], y_test, probability)
    plot_confusion(axes[1], y_test, predicted)
    fig.savefig(project / "images" / "results.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def wine_quality() -> None:
    project = ROOT / "projects" / "wine-quality"
    data = pd.read_csv(project / "data" / "processed" / "wine_quality_red_clean.csv")
    X = data.drop(columns="quality")
    y = data["quality"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = joblib.load(project / "artifacts" / "wine_quality_model.joblib")
    predicted = model.predict(X_test)
    error = y_test.to_numpy() - predicted
    mae = mean_absolute_error(y_test, predicted)
    score = r2_score(y_test, predicted)
    fig, axes = canvas(
        "Red wine | Sensory quality estimate",
        f"Random 20% holdout  •  MAE = {mae:.3f}  •  R² = {score:.3f}  •  Mean-baseline MAE = 0.705",
    )
    scatter, = axes[0].plot(y_test, predicted, "o", color=TEAL, alpha=0.68, markersize=5, markeredgecolor="white", markeredgewidth=0.35)
    limits = [2.7, 8.3]
    axes[0].plot(limits, limits, color=GOLD, linestyle="--", linewidth=2, label="Perfect prediction")
    axes[0].set(xlim=limits, ylim=limits, xlabel="Actual quality score", ylabel="Predicted quality score")
    axes[0].set_title("Predicted vs actual", loc="left", fontsize=14, pad=15)
    axes[0].grid(color=GRID, linewidth=0.8, alpha=0.8)
    axes[0].legend([scatter, axes[0].lines[-1]], ["Test samples", "Perfect prediction"], frameon=False, loc="upper left")

    axes[1].hist(error, bins=np.arange(-2.5, 2.76, 0.25), color=TEAL, alpha=0.88, edgecolor="white")
    axes[1].axvline(0, color=GOLD, linestyle="--", linewidth=2)
    axes[1].set(xlabel="Actual − predicted score", ylabel="Number of samples")
    axes[1].set_title("Prediction error distribution", loc="left", fontsize=14, pad=15)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    fig.savefig(project / "images" / "results.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    for project_name in ("bank-marketing", "adult-income", "wine-quality"):
        (ROOT / "projects" / project_name / "images").mkdir(parents=True, exist_ok=True)
    bank_marketing()
    adult_income()
    wine_quality()
    print("Created results.png for all three projects.")


if __name__ == "__main__":
    main()
