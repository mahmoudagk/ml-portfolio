"""Train and compare leakage-aware term-deposit response classifiers."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = PROJECT / "artifacts"
IMAGES = PROJECT / "images"
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
ZIP_PATH = RAW / "bank-marketing.zip"
TARGET = "y"
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "campaign", "pdays", "previous", "poutcome",
]
NAVY = "#18324B"
TEAL = "#139A9A"
GOLD = "#E9A33B"
CORAL = "#D96566"
MUTED = "#66788A"
GRID = "#DCE5ED"
BACKGROUND = "#F4F7FB"


def download_frame() -> pd.DataFrame:
    """Download/cache UCI's nested ZIP archive and read bank-full.csv."""
    RAW.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        request = Request(URL, headers={"User-Agent": "ml-portfolio/1.0"})
        with urlopen(request, timeout=60) as response:
            ZIP_PATH.write_bytes(response.read())
    with ZipFile(ZIP_PATH) as outer_archive:
        nested_name = next(name for name in outer_archive.namelist() if name.endswith("bank.zip"))
        with ZipFile(BytesIO(outer_archive.read(nested_name))) as archive:
            member = next(name for name in archive.namelist() if name.endswith("bank-full.csv"))
            with archive.open(member) as stream:
                return pd.read_csv(stream, sep=";")


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = frame.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]
    required = {
        "age", "job", "marital", "education", "default", "balance", "housing", "loan",
        "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y",
    }
    missing = sorted(required.difference(cleaned.columns))
    if missing:
        raise ValueError(f"UCI bank-full.csv is missing expected columns: {missing}")
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    cleaned[TARGET] = cleaned[TARGET].str.lower()
    raw_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    categorical = [column for column in cleaned.columns if column not in NUMERIC + [TARGET]]
    unknown_cells = int(cleaned[categorical].eq("unknown").sum().sum())
    cleaned[categorical] = cleaned[categorical].replace("unknown", np.nan)
    if not set(cleaned[TARGET].dropna().unique()).issubset({"yes", "no"}):
        raise ValueError("Unexpected labels in y; expected only yes and no.")
    missing_target_rows = int(cleaned[TARGET].isna().sum())
    cleaned = cleaned.dropna(subset=[TARGET]).reset_index(drop=True)
    quality = {
        "rows_raw": int(raw_rows),
        "rows_after_deduplication": int(len(cleaned)),
        "duplicate_rows_removed": int(raw_rows - len(cleaned)),
        "unknown_category_cells_converted_to_missing": unknown_cells,
        "rows_with_missing_target_removed": missing_target_rows,
    }
    return cleaned, quality


def make_pipeline(classifier, scale_numeric: bool = False) -> Pipeline:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                [column for column in FEATURES if column not in NUMERIC],
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def tune_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, force: bool = False) -> tuple[dict, pd.DataFrame]:
    tuning_path = ARTIFACTS / "hist_gradient_tuning.csv"
    if tuning_path.exists() and not force:
        cached = pd.read_csv(tuning_path).sort_values("rank_test_score")
        best = cached.iloc[0]
        params = {
            "max_iter": int(best["param_classifier__max_iter"]),
            "learning_rate": float(best["param_classifier__learning_rate"]),
            "max_leaf_nodes": int(best["param_classifier__max_leaf_nodes"]),
            "min_samples_leaf": int(best["param_classifier__min_samples_leaf"]),
            "l2_regularization": float(best["param_classifier__l2_regularization"]),
        }
        print("Reusing saved HGB search; pass --retune to repeat it.", flush=True)
        return params, cached

    search = RandomizedSearchCV(
        make_pipeline(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)),
        param_distributions={
            "classifier__max_iter": [100, 160, 220, 300],
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "classifier__max_leaf_nodes": [15, 31, 47, 63],
            "classifier__min_samples_leaf": [20, 40, 60, 80],
            "classifier__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
        n_iter=7,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    search.fit(X_train, y_train)
    params = {key.replace("classifier__", ""): value for key, value in search.best_params_.items()}
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    keep = [
        "rank_test_score", "mean_test_score", "std_test_score", "param_classifier__learning_rate",
        "param_classifier__max_iter", "param_classifier__max_leaf_nodes",
        "param_classifier__min_samples_leaf", "param_classifier__l2_regularization",
    ]
    results[keep].to_csv(tuning_path, index=False)
    return params, results[keep]


def binary_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "predicted_positive_rate": float(predictions.mean()),
    }


def threshold_sweep(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 3):
        metrics = binary_metrics(y_true, probabilities, float(threshold))
        rows.append(metrics)
    return pd.DataFrame(rows)


def plot_results(leaderboard: pd.DataFrame, y_test: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor(BACKGROUND)
    fig.text(0.04, 0.96, "Bank Marketing | term-deposit response", fontsize=22, fontweight="bold", color=NAVY, va="top")
    fig.text(0.04, 0.90, f"45,211 campaign records  ·  15 pre-call features  ·  selected: {leaderboard.iloc[0]['model']}", fontsize=11, color=MUTED, va="top")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.16, wspace=0.40)
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)

    ordered = leaderboard.sort_values("average_precision")
    axes[0].barh(ordered["model"], ordered["average_precision"], color=TEAL)
    axes[0].set_title("Validation average precision", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    axes[0].set_xlabel("Area under precision-recall curve")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="y", labelsize=8)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[1].plot(recall, precision, color=GOLD, linewidth=2.4)
    axes[1].axhline(float(y_test.mean()), linestyle="--", linewidth=1.4, color=MUTED, label=f"Positive rate = {y_test.mean():.1%}")
    axes[1].set_title("Held-out precision-recall curve", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(color=GRID, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)

    matrix = confusion_matrix(y_test, (probabilities >= threshold).astype(int), labels=[0, 1])
    axes[2].imshow(matrix, cmap="Blues")
    axes[2].set_title(f"Test confusion matrix · threshold {threshold:.2f}", loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axes[2].set_xticks([0, 1], labels=["No", "Yes"])
    axes[2].set_yticks([0, 1], labels=["No", "Yes"])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    for row in range(2):
        for col in range(2):
            axes[2].text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", fontsize=12,
                         color="white" if matrix[row, col] > matrix.max() / 2 else NAVY)
    fig.savefig(IMAGES / "results.png", dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_threshold_tradeoff(thresholds: pd.DataFrame, selected_threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.plot(thresholds["threshold"], thresholds["precision"], color=NAVY, label="Precision", linewidth=2)
    ax.plot(thresholds["threshold"], thresholds["recall"], color=TEAL, label="Recall", linewidth=2)
    ax.plot(thresholds["threshold"], thresholds["f1"], color=GOLD, label="F1", linewidth=2.6)
    ax.axvline(selected_threshold, color=CORAL, linestyle="--", label=f"Selected = {selected_threshold:.2f}")
    ax.set_title("Validation threshold trade-off", loc="left", fontsize=17, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Metric score")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGES / "threshold_tradeoff.png", dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame) -> None:
    top = importance.head(14).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.barh(top["feature"], top["importance_mean"], color=GOLD)
    ax.set_title("Validation permutation importance", loc="left", fontsize=17, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Average-precision decrease when shuffled")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMAGES / "feature_importance.png", dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def plot_calibration(y_test: pd.Series, probabilities: np.ndarray) -> None:
    fraction_positive, mean_predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor(BACKGROUND)
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.grid(color=GRID, linewidth=0.8)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=MUTED, label="Perfect calibration")
    axes[0].plot(mean_predicted, fraction_positive, marker="o", color=TEAL, linewidth=2, label="Selected model")
    axes[0].set_title("Calibration curve", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed subscription rate")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(probabilities, bins=np.linspace(0, 1, 21), color=GOLD, edgecolor="white")
    axes[1].set_title("Test probability distribution", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=12)
    axes[1].set_xlabel("Predicted subscription probability")
    axes[1].set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(IMAGES / "calibration.png", dpi=150, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models for term-deposit marketing response.")
    parser.add_argument("--retune", action="store_true", help="rerun the stratified 3-fold HGB hyperparameter search")
    args = parser.parse_args()
    for directory in [PROCESSED, ARTIFACTS, IMAGES]:
        directory.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

    source = download_frame()
    cleaned, quality = clean_data(source)
    cleaned.to_csv(PROCESSED / "bank_marketing_clean.csv", index=False)
    features = [column for column in cleaned.columns if column not in {TARGET, "duration"}]
    if features != FEATURES:
        raise ValueError(f"Unexpected source features or order: {features}")
    categorical = [column for column in features if column not in NUMERIC]
    X = cleaned[features].copy()
    for column in categorical:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    y = cleaned[TARGET].map({"no": 0, "yes": 1}).astype(int)
    if y.isna().any():
        raise ValueError("Target contains missing or unexpected labels.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
    )
    print(f"Rows={len(X):,}; features={len(features)}; train/validation/test={len(X_train):,}/{len(X_validation):,}/{len(X_test):,}", flush=True)

    hgb_params, tuning_results = tune_hist_gradient_boosting(X_train, y_train, force=args.retune)
    print(f"HGB best parameters: {hgb_params}", flush=True)
    candidates = [
        ("Prior baseline", make_pipeline(DummyClassifier(strategy="prior"))),
        ("Logistic regression", make_pipeline(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ), scale_numeric=True)),
        ("SGD log-loss", make_pipeline(SGDClassifier(
            loss="log_loss", class_weight="balanced", max_iter=2500, tol=1e-3, random_state=SEED
        ), scale_numeric=True)),
        ("Decision tree", make_pipeline(DecisionTreeClassifier(
            max_depth=14, min_samples_leaf=20, class_weight="balanced", random_state=SEED
        ))),
        ("Random forest", make_pipeline(RandomForestClassifier(
            n_estimators=220, max_depth=22, min_samples_leaf=5, max_features=0.8,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ))),
        ("Extra trees", make_pipeline(ExtraTreesClassifier(
            n_estimators=250, max_depth=24, min_samples_leaf=5, max_features=0.8,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        ))),
        ("Histogram gradient boosting tuned", make_pipeline(HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED, **hgb_params
        ))),
    ]

    leaderboard_rows = []
    fitted_models = {}
    validation_probabilities = {}
    for name, estimator in candidates:
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        probabilities = estimator.predict_proba(X_validation)[:, 1]
        row = {
            "model": name,
            "average_precision": float(average_precision_score(y_validation, probabilities)),
            "roc_auc": float(roc_auc_score(y_validation, probabilities)),
            "f1_at_0_5": float(f1_score(y_validation, probabilities >= 0.5, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_validation, probabilities >= 0.5, zero_division=0)),
            "fit_seconds": float(fit_seconds),
        }
        leaderboard_rows.append(row)
        fitted_models[name] = estimator
        validation_probabilities[name] = probabilities
        print(f"{name}: validation AP={row['average_precision']:.4f}, ROC-AUC={row['roc_auc']:.4f}", flush=True)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(ARTIFACTS / "validation_leaderboard.csv", index=False)
    selected_name = str(leaderboard.iloc[0]["model"])
    validation_model = fitted_models[selected_name]
    selected_validation_probabilities = validation_probabilities[selected_name]

    threshold_table = threshold_sweep(y_validation, selected_validation_probabilities)
    threshold_table.to_csv(ARTIFACTS / "threshold_analysis.csv", index=False)
    best_threshold_row = threshold_table.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    selected_threshold = float(best_threshold_row["threshold"])
    importance_result = permutation_importance(
        validation_model,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": features,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(ARTIFACTS / "permutation_importance.csv", index=False)

    final_model = fitted_models[selected_name]
    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    final_model.fit(X_train_validation, y_train_validation)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_default_metrics = binary_metrics(y_test, test_probabilities, 0.5)
    test_selected_metrics = binary_metrics(y_test, test_probabilities, selected_threshold)
    test_predictions = (test_probabilities >= selected_threshold).astype(int)
    confusion = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    pd.DataFrame(confusion, index=["actual_no", "actual_yes"], columns=["predicted_no", "predicted_yes"]).to_csv(
        ARTIFACTS / "confusion_matrix.csv"
    )
    report = classification_report(y_test, test_predictions, target_names=["no", "yes"], output_dict=True, zero_division=0)
    (ARTIFACTS / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({
        "row_index": X_test.index.to_numpy(),
        "actual": y_test.to_numpy(),
        "probability_yes": test_probabilities,
        "predicted_at_selected_threshold": test_predictions,
    }).to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    joblib.dump({"model": final_model, "threshold": selected_threshold, "features": features},
                ARTIFACTS / "bank_marketing_model.joblib", compress=3)

    validation_metrics = binary_metrics(y_validation, selected_validation_probabilities, selected_threshold)
    metrics = {
        "task": "binary classification: term-deposit subscription",
        "source_dataset": "UCI Bank Marketing bank-full.csv",
        **quality,
        "positive_rate": float(y.mean()),
        "features_used": features,
        "categorical_features": categorical,
        "excluded_features": {
            "duration": "Known only after a call; excluded for a pre-call prediction scenario."
        },
        "cleaning_notes": {
            "unknown_category": "Converted to missing; categorical imputation is fitted inside each model pipeline.",
            "duplicates": "Exact duplicate rows removed before splitting.",
        },
        "split": {
            "method": "stratified 60% train / 20% validation / 20% test",
            "seed": SEED,
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
        },
        "tuning": {
            "model": "HistGradientBoostingClassifier",
            "method": "7 randomized parameter configurations, stratified 3-fold cross-validation",
            "selection_metric": "average precision",
            "best_cv_average_precision": float(tuning_results.iloc[0]["mean_test_score"]),
            "best_parameters": hgb_params,
        },
        "models_compared": leaderboard["model"].tolist(),
        "validation_leaderboard_metric": "average_precision",
        "selected_model": selected_name,
        "validation_metrics_at_selected_threshold": validation_metrics,
        "threshold_selection": {
            "method": "maximize validation F1 on thresholds 0.05 through 0.95",
            "threshold": selected_threshold,
        },
        "test_metrics_at_selected_threshold": test_selected_metrics,
        "test_metrics_at_default_threshold_0_5": test_default_metrics,
        "confusion_matrix_at_selected_threshold": confusion.tolist(),
        "target_counts": {"no": int((y == 0).sum()), "yes": int((y == 1).sum())},
        "test_target_counts": {"no": int((y_test == 0).sum()), "yes": int((y_test == 1).sum())},
        "model_file": "artifacts/bank_marketing_model.joblib",
        "seed": SEED,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_results(leaderboard, y_test, test_probabilities, selected_threshold)
    plot_threshold_tradeoff(threshold_table, selected_threshold)
    plot_importance(importance)
    plot_calibration(y_test, test_probabilities)
    print(
        f"Selected {selected_name}; test AP={test_selected_metrics['average_precision']:.4f}, "
        f"F1={test_selected_metrics['f1']:.4f} at threshold={selected_threshold:.2f}",
        flush=True,
    )
    print(f"Artifacts saved under {ARTIFACTS}", flush=True)


if __name__ == "__main__":
    main()
