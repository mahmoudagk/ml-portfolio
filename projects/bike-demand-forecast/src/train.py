"""Hourly bike-rental demand forecasting with a chronological evaluation."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


SEED = 42
PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
ARTIFACTS = PROJECT / "artifacts"
IMAGES = PROJECT / "images"
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
ZIP_PATH = RAW / "bike-sharing-dataset.zip"
CSV_PATH = RAW / "hour.csv"
TARGET = "cnt"

NUMERIC_FEATURES = [
    "year_index",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "annual_sin",
    "annual_cos",
]
CATEGORICAL_FEATURES = [
    "season",
    "month",
    "hour",
    "weekday",
    "weather",
    "holiday",
    "working_day",
    "weekend",
    "year",
]
NAVY = "#18324B"
TEAL = "#139A9A"
GOLD = "#E9A33B"
MUTED = "#66788A"
GRID = "#DCE5ED"


def get_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """Download the official archive, validate it, and build safe features."""
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        if not ZIP_PATH.exists():
            request = urllib.request.Request(DATA_URL, headers={"User-Agent": "ml-portfolio/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, ZIP_PATH.open("wb") as output:
                output.write(response.read())
        with zipfile.ZipFile(ZIP_PATH) as archive:
            member = next(name for name in archive.namelist() if name.endswith("hour.csv"))
            CSV_PATH.write_bytes(archive.read(member))

    source = pd.read_csv(CSV_PATH)
    required = {
        "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
        "registered", "cnt",
    }
    missing_columns = sorted(required.difference(source.columns))
    if missing_columns:
        raise ValueError(f"UCI hour.csv is missing expected columns: {missing_columns}")

    initial_rows = len(source)
    source = source.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(source)
    source["datetime"] = pd.to_datetime(source["dteday"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(source["hr"], errors="coerce"), unit="h"
    )
    source[TARGET] = pd.to_numeric(source[TARGET], errors="coerce")
    invalid_rows = int(source[["datetime", TARGET]].isna().any(axis=1).sum())
    source = source.dropna(subset=["datetime", TARGET]).sort_values(["datetime", "instant"]).reset_index(drop=True)
    if source.empty or source[TARGET].lt(0).any():
        raise ValueError("The cleaned target is empty or contains negative rental counts.")
    if not (source["cnt"].astype(int) == source["casual"].astype(int) + source["registered"].astype(int)).all():
        raise ValueError("The source target does not equal casual + registered counts.")

    dt = source["datetime"]
    hour = dt.dt.hour
    weekday = dt.dt.day_of_week
    day_of_year = dt.dt.day_of_year
    year_length = np.where(dt.dt.is_leap_year, 366, 365)
    hour_angle = 2 * np.pi * hour / 24
    weekday_angle = 2 * np.pi * weekday / 7
    year_angle = 2 * np.pi * (day_of_year - 1) / year_length

    features = pd.DataFrame(index=source.index)
    features["year_index"] = dt.dt.year - dt.dt.year.min()
    features["temp"] = pd.to_numeric(source["temp"], errors="coerce")
    features["atemp"] = pd.to_numeric(source["atemp"], errors="coerce")
    features["humidity"] = pd.to_numeric(source["hum"], errors="coerce")
    features["windspeed"] = pd.to_numeric(source["windspeed"], errors="coerce")
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    features["weekday_sin"] = np.sin(weekday_angle)
    features["weekday_cos"] = np.cos(weekday_angle)
    features["annual_sin"] = np.sin(year_angle)
    features["annual_cos"] = np.cos(year_angle)
    features["season"] = source["season"].astype("Int64").astype("string")
    features["month"] = dt.dt.month.astype("string")
    features["hour"] = hour.astype("string")
    features["weekday"] = weekday.astype("string")
    features["weather"] = source["weathersit"].astype("Int64").astype("string")
    features["holiday"] = source["holiday"].astype("Int64").astype("string")
    features["working_day"] = source["workingday"].astype("Int64").astype("string")
    features["weekend"] = (weekday >= 5).astype("string")
    features["year"] = dt.dt.year.astype("string")
    target = source[TARGET].astype(float)
    timestamps = source["datetime"]

    clean = features.copy()
    clean.insert(0, "datetime", timestamps.to_numpy())
    clean[TARGET] = target.to_numpy()
    clean.to_csv(PROCESSED / "bike_hourly_features.csv", index=False)

    quality = {
        "source_rows": int(initial_rows),
        "clean_rows": int(len(source)),
        "exact_duplicates_removed": int(duplicates_removed),
        "rows_missing_timestamp_or_target_removed": int(invalid_rows),
        "source_model_signal_count": 12,
        "engineered_feature_count": int(features.shape[1] - 12),
        "model_feature_count": int(features.shape[1]),
        "target_components_check": "cnt equals casual + registered; component columns excluded from model",
        "date_start": timestamps.min().isoformat(),
        "date_end": timestamps.max().isoformat(),
    }
    return features, target, timestamps, quality


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def pipeline(regressor, scale_numeric: bool = False) -> Pipeline:
    return Pipeline([
        ("preprocess", make_preprocessor(scale_numeric=scale_numeric)),
        ("regressor", regressor),
    ])


def score_predictions(actual: pd.Series, prediction: np.ndarray) -> dict:
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0)
    actual_array = np.asarray(actual, dtype=float)
    absolute_error = np.abs(actual_array - prediction)
    return {
        "mae": float(mean_absolute_error(actual_array, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, prediction))),
        "r2": float(r2_score(actual_array, prediction)),
        "wape": float(absolute_error.sum() / max(actual_array.sum(), 1.0)),
    }


def clipped_negative_mae(estimator, X, y) -> float:
    return -mean_absolute_error(y, np.maximum(estimator.predict(X), 0))


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, force: bool = False
) -> tuple[dict, pd.DataFrame]:
    tuning_path = ARTIFACTS / "hist_gradient_tuning.csv"
    if tuning_path.exists() and not force:
        cached = pd.read_csv(tuning_path).sort_values("rank_test_score")
        best = cached.iloc[0]
        parameters = {
            "loss": str(best["param_regressor__loss"]),
            "learning_rate": float(best["param_regressor__learning_rate"]),
            "max_iter": int(best["param_regressor__max_iter"]),
            "max_leaf_nodes": int(best["param_regressor__max_leaf_nodes"]),
            "min_samples_leaf": int(best["param_regressor__min_samples_leaf"]),
            "l2_regularization": float(best["param_regressor__l2_regularization"]),
        }
        print("Reusing saved HistGradientBoosting search; pass --retune to repeat it.", flush=True)
        return parameters, cached

    search = RandomizedSearchCV(
        pipeline(HistGradientBoostingRegressor(random_state=SEED)),
        param_distributions={
            "regressor__loss": ["squared_error", "absolute_error", "poisson"],
            "regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "regressor__max_iter": [100, 150, 220, 300],
            "regressor__max_leaf_nodes": [15, 31, 47, 63],
            "regressor__min_samples_leaf": [15, 25, 40, 60],
            "regressor__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        },
        n_iter=7,
        scoring=clipped_negative_mae,
        cv=TimeSeriesSplit(n_splits=3),
        random_state=SEED,
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    search.fit(X_train, y_train)
    parameters = {key.replace("regressor__", ""): value for key, value in search.best_params_.items()}
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    keep = [
        "rank_test_score", "mean_test_score", "std_test_score", "param_regressor__loss",
        "param_regressor__learning_rate", "param_regressor__max_iter",
        "param_regressor__max_leaf_nodes", "param_regressor__min_samples_leaf",
        "param_regressor__l2_regularization",
    ]
    results[keep].to_csv(tuning_path, index=False)
    return parameters, results[keep]


def plot_dashboard(leaderboard: pd.DataFrame, predictions: pd.DataFrame, importance: pd.DataFrame, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=False)
    fig.patch.set_facecolor("#F4F7FB")
    fig.text(0.04, 0.96, "Bike demand | chronological forecasting", fontsize=22, fontweight="bold", color=NAVY, va="top")
    fig.text(0.04, 0.90, f"17K hourly records  •  20 model inputs  •  selected: {metrics['selected_model']}", fontsize=11, color=MUTED, va="top")
    fig.text(0.04, 0.035, f"Held-out test MAE = {metrics['test_metrics']['mae']:.1f} bikes  •  WAPE = {metrics['test_metrics']['wape']:.1%}", fontsize=10, color=MUTED)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.77, bottom=0.16, wspace=0.42)
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)

    ordered = leaderboard.sort_values("mae", ascending=False)
    label_map = {
        "Histogram gradient boosting tuned": "Hist. gradient boost",
        "Gradient boosting": "Gradient boost",
        "Random forest": "Random forest",
        "Extra trees": "Extra trees",
        "Decision tree": "Decision tree",
        "Ridge regression": "Ridge",
        "Mean baseline": "Mean baseline",
    }
    axes[0].barh(ordered["model"].replace(label_map), ordered["mae"], color=TEAL)
    axes[0].set_title("Validation MAE (lower is better)", loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axes[0].set_xlabel("Mean absolute error (bikes)")
    axes[0].grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)
    axes[0].tick_params(axis="y", labelsize=8)

    sample = predictions.head(24 * 7)
    axes[1].plot(sample["datetime"], sample["actual"], color=NAVY, linewidth=1.5, label="Actual")
    axes[1].plot(sample["datetime"], sample["predicted"], color=GOLD, linewidth=1.4, label="Predicted")
    axes[1].set_title("First week of test period", loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axes[1].set_ylabel("Rentals per hour")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)

    top = importance.head(12).sort_values("importance_mean", ascending=True)
    axes[2].barh(top["feature"], top["importance_mean"], color=GOLD)
    axes[2].set_title("Validation permutation importance", loc="left", fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axes[2].set_xlabel("MAE increase when shuffled")
    axes[2].grid(axis="x", color=GRID, linewidth=0.8)
    axes[2].set_axisbelow(True)
    axes[2].tick_params(axis="y", labelsize=8)
    fig.savefig(IMAGES / "results.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast hourly bike rentals using a chronological holdout.")
    parser.add_argument("--retune", action="store_true", help="rerun the 3-fold time-series hyperparameter search")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    X, y, timestamps, quality = get_data()
    train_end = int(len(X) * 0.70)
    validation_end = int(len(X) * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_validation, y_validation = X.iloc[train_end:validation_end], y.iloc[train_end:validation_end]
    X_test, y_test = X.iloc[validation_end:], y.iloc[validation_end:]
    test_timestamps = timestamps.iloc[validation_end:]
    if not (timestamps.iloc[:train_end].max() < timestamps.iloc[train_end:validation_end].min()
            and timestamps.iloc[train_end:validation_end].max() < test_timestamps.min()):
        raise ValueError("Chronological train, validation, and test periods overlap.")

    print(
        f"Rows={len(X):,} | features={X.shape[1]} | "
        f"train/validation/test={len(X_train):,}/{len(X_validation):,}/{len(X_test):,}",
        flush=True,
    )
    best_hgb_parameters, tuning_results = tune_hist_gradient_boosting(X_train, y_train, force=args.retune)
    print(f"Best HistGradientBoosting parameters: {best_hgb_parameters}", flush=True)

    models = [
        ("Mean baseline", pipeline(DummyRegressor(strategy="mean"))),
        ("Ridge regression", pipeline(Ridge(alpha=10.0), scale_numeric=True)),
        ("Decision tree", pipeline(DecisionTreeRegressor(max_depth=24, min_samples_leaf=8, random_state=SEED))),
        ("Random forest", pipeline(RandomForestRegressor(
            n_estimators=220, max_depth=24, min_samples_leaf=3, max_features=0.9,
            n_jobs=-1, random_state=SEED,
        ))),
        ("Extra trees", pipeline(ExtraTreesRegressor(
            n_estimators=250, max_depth=25, min_samples_leaf=3, max_features=0.9,
            n_jobs=-1, random_state=SEED,
        ))),
        ("Gradient boosting", pipeline(GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.06, max_depth=3,
            min_samples_leaf=10, loss="huber", random_state=SEED,
        ))),
        ("Histogram gradient boosting tuned", pipeline(
            HistGradientBoostingRegressor(random_state=SEED, **best_hgb_parameters)
        )),
    ]

    leaderboard_rows = []
    validation_predictions = {}
    fitted_validation = {}
    for name, estimator in models:
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        prediction = np.maximum(estimator.predict(X_validation), 0)
        row = {"model": name, **score_predictions(y_validation, prediction), "fit_seconds": fit_seconds}
        leaderboard_rows.append(row)
        validation_predictions[name] = prediction
        fitted_validation[name] = estimator
        print(f"{name}: validation MAE={row['mae']:.2f}, RMSE={row['rmse']:.2f}, R2={row['r2']:.3f}", flush=True)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mae").reset_index(drop=True)
    leaderboard.to_csv(ARTIFACTS / "validation_leaderboard.csv", index=False)
    selected_name = str(leaderboard.loc[0, "model"])
    selected_validation_model = fitted_validation[selected_name]
    importance_result = permutation_importance(
        selected_validation_model,
        X_validation,
        y_validation,
        scoring=clipped_negative_mae,
        n_repeats=6,
        random_state=SEED,
        n_jobs=-1,
    )
    # Permutation importance is measured on the original input columns, before one-hot expansion.
    feature_names = X_validation.columns
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(ARTIFACTS / "permutation_importance.csv", index=False)

    final_model = next(estimator for name, estimator in models if name == selected_name)
    X_train_validation = pd.concat([X_train, X_validation], axis=0)
    y_train_validation = pd.concat([y_train, y_validation], axis=0)
    final_model.fit(X_train_validation, y_train_validation)
    test_prediction = np.maximum(final_model.predict(X_test), 0)
    test_metrics = score_predictions(y_test, test_prediction)
    prediction_frame = pd.DataFrame({
        "datetime": test_timestamps.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": test_prediction,
    })
    prediction_frame.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    joblib.dump(final_model, ARTIFACTS / "bike_demand_model.joblib", compress=3)

    metrics = {
        "task": "hourly bike rental demand regression",
        "source_dataset": "UCI Bike Sharing Dataset, hourly file",
        "rows": int(len(X)),
        "source_feature_count": 12,
        "engineered_feature_count": int(X.shape[1] - 12),
        "model_feature_count": int(X.shape[1]),
        "data_quality": quality,
        "split": {
            "method": "chronological; first 70% train, next 15% validation, final 15% test",
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "train_end": timestamps.iloc[train_end - 1].isoformat(),
            "validation_end": timestamps.iloc[validation_end - 1].isoformat(),
            "test_start": test_timestamps.min().isoformat(),
            "test_end": test_timestamps.max().isoformat(),
        },
        "tuning": {
            "model": "HistGradientBoostingRegressor",
            "method": "RandomizedSearchCV with 3 expanding-window TimeSeriesSplit folds",
            "scoring": "negative MAE with predictions clipped to zero",
            "iterations": int(len(tuning_results)),
            "best_parameters": best_hgb_parameters,
            "best_cv_mae": float(-tuning_results.iloc[0]["mean_test_score"]),
        },
        "selected_model": selected_name,
        "selection_metric": "validation MAE",
        "validation_metrics": leaderboard.iloc[0].drop(labels=["model", "fit_seconds"]).to_dict(),
        "test_metrics": test_metrics,
        "model_file": "artifacts/bike_demand_model.joblib",
        "target_leakage_controls": [
            "dropped casual and registered component counts because cnt is their sum",
            "dropped instant source row ID",
            "used a chronological split rather than a random row split",
            "refit only after selecting on validation; scored the held-out test once",
        ],
        "seed": SEED,
    }
    (ARTIFACTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_dashboard(leaderboard, prediction_frame, importance, metrics)
    print(f"Selected: {selected_name} | test MAE={test_metrics['mae']:.2f} | test R2={test_metrics['r2']:.3f}", flush=True)
    print(f"Artifacts saved in {ARTIFACTS}", flush=True)


if __name__ == "__main__":
    main()
