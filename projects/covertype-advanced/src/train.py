from __future__ import annotations

from datetime import datetime, timezone
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile
import argparse
import ast
import gc
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
DATA = PROJECT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = PROJECT / "artifacts"
IMAGES = PROJECT / "images"
URL = "https://archive.ics.uci.edu/static/public/31/covertype.zip"
SEED = 42
TARGET = "Cover_Type"
CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}
NUMERIC_SOURCE = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL = [f"Soil_Type{i}" for i in range(1, 41)]
RAW_COLUMNS = NUMERIC_SOURCE + WILDERNESS + SOIL + [TARGET]
NUMERIC_DTYPES = {column: np.int16 for column in NUMERIC_SOURCE}
NUMERIC_DTYPES.update({column: np.uint8 for column in WILDERNESS + SOIL + [TARGET]})
LABELS = sorted(CLASS_NAMES)
NAVY, TEAL, GOLD, MUTED, GRID = "#17324D", "#159A9C", "#E6A23C", "#66788A", "#DDE6EE"


def download_raw_frame() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    archive_path = RAW / "covertype.zip"
    if not archive_path.exists():
        request = Request(URL, headers={"User-Agent": "ml-portfolio/1.0"})
        with urlopen(request, timeout=120) as response:
            archive_path.write_bytes(response.read())
    with ZipFile(archive_path) as archive:
        member = next(
            name for name in archive.namelist()
            if Path(name).name.lower() in {"covtype.data.gz", "covtype.data"}
        )
        content = archive.read(member)
    stream = GzipFile(fileobj=BytesIO(content)) if member.lower().endswith(".gz") else BytesIO(content)
    return pd.read_csv(stream, header=None, names=RAW_COLUMNS, dtype=NUMERIC_DTYPES)


def validate_and_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError(f"Expected {len(RAW_COLUMNS)} source columns; found {len(raw.columns)}.")
    if raw.isna().any().any():
        raise ValueError("Unexpected missing cells found in the UCI source data.")
    if not set(raw[TARGET].unique()).issubset(LABELS):
        raise ValueError("Unexpected target labels; expected cover types 1 through 7.")
    for column in WILDERNESS + SOIL:
        if not raw[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0/1 values.")
    if not raw[WILDERNESS].sum(axis=1).eq(1).all():
        raise ValueError("Each row must belong to exactly one wilderness area.")
    if not raw[SOIL].sum(axis=1).eq(1).all():
        raise ValueError("Each row must belong to exactly one soil type.")
    if not raw["Aspect"].between(0, 360).all():
        raise ValueError("Aspect values must be in the 0–360 degree range.")
    hillshade = raw[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]]
    if not hillshade.apply(lambda col: col.between(0, 255).all()).all():
        raise ValueError("Hillshade values must be in the 0–255 range.")

    data = raw.copy()
    aspect_radians = np.deg2rad(data["Aspect"].astype(np.float32))
    data["Aspect_Sin"] = np.sin(aspect_radians).astype(np.float32)
    data["Aspect_Cos"] = np.cos(aspect_radians).astype(np.float32)
    data["Hydrology_Distance_3D"] = np.hypot(
        data["Horizontal_Distance_To_Hydrology"].astype(np.float32),
        data["Vertical_Distance_To_Hydrology"].astype(np.float32),
    ).astype(np.float32)
    shade = data[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].astype(np.float32)
    data["Hillshade_Mean"] = shade.mean(axis=1).astype(np.float32)
    data["Hillshade_Range"] = (shade.max(axis=1) - shade.min(axis=1)).astype(np.float32)
    data["Log_Roadway_Distance"] = np.log1p(data["Horizontal_Distance_To_Roadways"].astype(np.float32))
    data["Log_Fire_Distance"] = np.log1p(data["Horizontal_Distance_To_Fire_Points"].astype(np.float32))
    data["Log_Hydrology_Distance"] = np.log1p(data["Hydrology_Distance_3D"])
    return data


def get_dataset() -> pd.DataFrame:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED / "covertype_clean.csv.gz"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    raw = download_raw_frame()
    clean = validate_and_engineer(raw)
    clean.to_csv(clean_path, index=False, compression="gzip")
    return clean


def make_scaled_pipeline(estimator, numeric_features: list[str]) -> Pipeline:
    scaler = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("scale", scaler), ("model", estimator)])


def tune_hist_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series, force: bool = False):
    tune_rows = min(15_000, len(X_train))
    tuning_path = ARTIFACTS / "hist_gradient_tuning.csv"
    if tuning_path.exists() and not force:
        cached = pd.read_csv(tuning_path)
        if len(cached) and {"params", "mean_test_score", "rank_test_score"}.issubset(cached.columns):
            best = cached.loc[cached["rank_test_score"].idxmin()]
            return ast.literal_eval(best["params"]), float(best["mean_test_score"]), int(tune_rows), True
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=tune_rows, random_state=SEED, stratify=y_train
    )
    search = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(min_samples_leaf=20, early_stopping=True, random_state=SEED),
        param_distributions={
            "learning_rate": [0.05, 0.08, 0.1],
            "max_iter": [80, 120],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0, 5.0],
        },
        n_iter=3,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        refit=True,
        random_state=SEED,
        n_jobs=1,
        verbose=0,
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_tune)
    search.fit(X_tune, y_tune, sample_weight=weights)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(tuning_path, index=False)
    return search.best_params_, float(search.best_score_), int(tune_rows), False


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    started = perf_counter()
    prediction = model.predict(X)
    predict_seconds = perf_counter() - started
    probability = model.predict_proba(X)
    metrics = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "weighted_f1": float(f1_score(y, prediction, average="weighted")),
        "macro_precision": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "macro_roc_auc_ovr": float(roc_auc_score(y, probability, labels=LABELS, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y, probability, labels=LABELS)),
        "predict_seconds": float(predict_seconds),
    }
    return metrics, prediction, probability


def plot_dashboard(leaderboard: pd.DataFrame, matrix: np.ndarray, importance: pd.DataFrame, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), constrained_layout=False)
    fig.patch.set_facecolor("#F4F7FB")
    fig.text(0.04, 0.95, "Covertype | 7-class model benchmark", fontsize=23, fontweight="bold", color=NAVY, va="top")
    fig.text(0.04, 0.895, f"581,012 terrain cells  •  62 engineered features  •  selected: {metrics['selected_model']}", fontsize=12, color=MUTED, va="top")
    fig.text(0.04, 0.045, f"Untouched test macro-F1 = {metrics['macro_f1']:.3f}  •  Balanced accuracy = {metrics['balanced_accuracy']:.3f}", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.78, bottom=0.14, wspace=0.35)
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)

    ordered = leaderboard.sort_values("macro_f1", ascending=True)
    display_names = {
        "HistGradientBoosting tuned": "Hist. gradient boost",
        "Logistic regression": "Logistic reg.",
        "Prior baseline": "Prior baseline",
    }
    axes[0].barh(ordered["model"].replace(display_names), ordered["macro_f1"], color=TEAL)
    axes[0].set_title("Validation macro-F1", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=15)
    axes[0].set_xlabel("Macro-F1")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="y", labelsize=9)

    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_title("Final test confusion matrix", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=15)
    short_names = ["Spruce", "Lodgepole", "Ponderosa", "Cottonwood", "Aspen", "Douglas-fir", "Krummholz"]
    axes[1].set_xticks(range(7), labels=short_names, rotation=55, ha="right", fontsize=8)
    axes[1].set_yticks(range(7), labels=short_names, fontsize=8)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("Actual class")
    midpoint = matrix.max() / 2
    for row in range(7):
        for column in range(7):
            color = "white" if matrix[row, column] > midpoint else NAVY
            axes[1].text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=7, color=color)

    top = importance.head(14).sort_values("importance_mean", ascending=True)
    axes[2].barh(top["feature"], top["importance_mean"], color=GOLD)
    axes[2].set_title("Permutation importance", loc="left", fontsize=14, fontweight="bold", color=NAVY, pad=15)
    axes[2].set_xlabel("Macro-F1 decrease")
    axes[2].grid(axis="x", color=GRID, linewidth=0.8)
    axes[2].set_axisbelow(True)
    axes[2].tick_params(axis="y", labelsize=8)
    fig.savefig(IMAGES / "results.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare models on the UCI Covertype dataset.")
    parser.add_argument("--retune", action="store_true", help="rerun the 3-fold hyperparameter search instead of reusing saved tuning results")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data = get_dataset()
    features = [column for column in data.columns if column != TARGET]
    numeric_features = NUMERIC_SOURCE + [
        "Aspect_Sin", "Aspect_Cos", "Hydrology_Distance_3D", "Hillshade_Mean",
        "Hillshade_Range", "Log_Roadway_Distance", "Log_Fire_Distance", "Log_Hydrology_Distance",
    ]
    X = data[features].astype(np.float32)
    y = data[TARGET].astype(np.uint8)
    del data
    gc.collect()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    validation_fraction = 0.15 / 0.85
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val, y_train_val, test_size=validation_fraction, random_state=SEED, stratify=y_train_val
    )
    del X_train_val, y_train_val
    gc.collect()

    print(f"Rows: {len(X):,}; features: {X.shape[1]}; train/validation/test: {len(X_train):,}/{len(X_validation):,}/{len(X_test):,}", flush=True)
    best_hgb_params, hgb_cv_macro_f1, tuning_rows, tuning_reused = tune_hist_gradient_boosting(X_train, y_train, force=args.retune)
    print(f"HistGradientBoosting CV macro-F1: {hgb_cv_macro_f1:.4f}; best parameters: {best_hgb_params}", flush=True)
    benchmark_rows = min(150_000, len(X_train))
    X_benchmark, _, y_benchmark, _ = train_test_split(
        X_train, y_train, train_size=benchmark_rows, random_state=SEED, stratify=y_train
    )
    print(f"Benchmarking each model on the same stratified sample of {len(X_benchmark):,} training rows.", flush=True)

    candidates = [
        ("Prior baseline", DummyClassifier(strategy="prior"), False),
        ("Logistic regression", make_scaled_pipeline(
            LogisticRegression(solver="saga", max_iter=60, tol=5e-3, class_weight="balanced", random_state=SEED), numeric_features,
        ), False),
        ("SGD log-loss", make_scaled_pipeline(
            SGDClassifier(loss="log_loss", max_iter=40, tol=1e-3, class_weight="balanced", early_stopping=True, random_state=SEED, n_jobs=-1), numeric_features,
        ), False),
        ("Decision tree", DecisionTreeClassifier(max_depth=28, min_samples_leaf=5, class_weight="balanced", random_state=SEED), False),
        ("Random forest", RandomForestClassifier(
            n_estimators=80, max_depth=24, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", max_samples=0.7, n_jobs=-1, random_state=SEED,
        ), False),
        ("Extra trees", ExtraTreesClassifier(
            n_estimators=100, max_depth=28, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", max_samples=0.7, bootstrap=True, n_jobs=-1, random_state=SEED,
        ), False),
        ("HistGradientBoosting tuned", HistGradientBoostingClassifier(
            **best_hgb_params, min_samples_leaf=20, early_stopping=True, random_state=SEED,
        ), True),
    ]

    rows = []
    selected_name = None
    selected_template = None
    best_validation_f1 = -1.0
    for name, estimator, needs_sample_weight in candidates:
        print(f"Training {name}...", flush=True)
        model = clone(estimator)
        started = perf_counter()
        if needs_sample_weight:
            model.fit(X_benchmark, y_benchmark, sample_weight=compute_sample_weight(class_weight="balanced", y=y_benchmark))
        else:
            model.fit(X_benchmark, y_benchmark)
        fit_seconds = perf_counter() - started
        scores, _, _ = evaluate_model(model, X_validation, y_validation)
        scores.update({"model": name, "fit_seconds": float(fit_seconds)})
        rows.append(scores)
        print(f"  macro-F1={scores['macro_f1']:.4f}, balanced-accuracy={scores['balanced_accuracy']:.4f}, fit={fit_seconds:.1f}s", flush=True)
        if scores["macro_f1"] > best_validation_f1:
            selected_name = name
            selected_template = clone(estimator)
            best_validation_f1 = scores["macro_f1"]
        del model
        gc.collect()

    leaderboard = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(ARTIFACTS / "validation_leaderboard.csv", index=False)
    print(f"Selected on validation macro-F1: {selected_name}", flush=True)

    X_refit = pd.concat([X_train, X_validation], ignore_index=True)
    y_refit = pd.concat([y_train, y_validation], ignore_index=True)
    final_refit_rows = min(200_000, len(X_refit))
    X_final_train, _, y_final_train, _ = train_test_split(
        X_refit, y_refit, train_size=final_refit_rows, random_state=SEED, stratify=y_refit
    )
    final_model = clone(selected_template)
    if selected_name == "HistGradientBoosting tuned":
        final_model.fit(X_final_train, y_final_train, sample_weight=compute_sample_weight(class_weight="balanced", y=y_final_train))
    else:
        final_model.fit(X_final_train, y_final_train)
    test_scores, test_predictions, _ = evaluate_model(final_model, X_test, y_test)

    report = classification_report(
        y_test, test_predictions, labels=LABELS, target_names=[CLASS_NAMES[label] for label in LABELS], output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_test, test_predictions, labels=LABELS)
    importance_rows = min(3_000, len(X_test))
    X_importance, _, y_importance, _ = train_test_split(
        X_test, y_test, train_size=importance_rows, stratify=y_test, random_state=SEED
    )
    permutation = permutation_importance(
        final_model, X_importance, y_importance, scoring="f1_macro", n_repeats=3, random_state=SEED, n_jobs=1,
    )
    importance = pd.DataFrame(
        {"feature": features, "importance_mean": permutation.importances_mean, "importance_std": permutation.importances_std}
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    importance.to_csv(ARTIFACTS / "permutation_importance.csv", index=False)
    pd.DataFrame(matrix, index=[CLASS_NAMES[value] for value in LABELS], columns=[CLASS_NAMES[value] for value in LABELS]).to_csv(
        ARTIFACTS / "confusion_matrix.csv"
    )
    (ARTIFACTS / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(final_model, ARTIFACTS / "covertype_final_model.joblib", compress=3)

    metrics = {
        "task": "7-class classification",
        "source_dataset": "UCI Covertype (dataset id 31)",
        "rows": int(len(X)),
        "source_feature_count": 54,
        "engineered_feature_count": int(X.shape[1] - 54),
        "model_feature_count": int(X.shape[1]),
        "split_rows": {"train": int(len(X_train)), "validation": int(len(X_validation)), "test": int(len(X_test))},
        "model_comparison_train_rows": int(len(X_benchmark)),
        "final_refit_rows": int(len(X_final_train)),
        "split_seed": SEED,
        "tuning": {"method": "RandomizedSearchCV", "cv": "3-fold stratified", "scoring": "macro-F1", "sample_rows": tuning_rows, "best_cv_macro_f1": hgb_cv_macro_f1, "best_parameters": best_hgb_params},
        "tuning_results_reused": bool(tuning_reused),
        "selected_model": selected_name,
        "selection_metric": "validation macro-F1",
        "validation_macro_f1": float(best_validation_f1),
        "test_metrics": test_scores,
        "class_names": CLASS_NAMES,
        "class_distribution": {str(label): int((y == label).sum()) for label in LABELS},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_dashboard(leaderboard, matrix, importance, {"selected_model": selected_name, **test_scores})
    print(json.dumps({"selected_model": selected_name, **test_scores}, indent=2), flush=True)
    print(f"Artifacts saved to {ARTIFACTS}", flush=True)


if __name__ == "__main__":
    main()
