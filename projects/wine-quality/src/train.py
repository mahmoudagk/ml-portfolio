from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
PROCESSED = DATA / "processed"
ARTIFACTS = PROJECT / "artifacts"
URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
TARGET = "quality"


def download_frame() -> pd.DataFrame:
    request = Request(URL, headers={"User-Agent": "ml-portfolio/1.0"})
    with urlopen(request, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))
    member = next(name for name in archive.namelist() if name.endswith("winequality-red.csv"))
    with archive.open(member) as stream:
        return pd.read_csv(stream, sep=";")


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [column.strip().lower().replace(" ", "_") for column in cleaned.columns]
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.drop_duplicates().dropna().reset_index(drop=True)
    cleaned = cleaned[cleaned[TARGET].between(0, 10)].reset_index(drop=True)
    return cleaned


def main() -> None:
    frame = download_frame()
    cleaned = clean_data(frame)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED / "wine_quality_red_clean.csv", index=False)

    X = cleaned.drop(columns=TARGET)
    y = cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    baseline = np.full(len(y_test), y_train.mean())
    metrics = {
        "task": "regression",
        "rows_raw": int(len(frame)),
        "rows_after_cleaning": int(len(cleaned)),
        "features_used": list(X.columns),
        "target_range": [float(cleaned[TARGET].min()), float(cleaned[TARGET].max())],
        "test_size": 0.2,
        "random_state": 42,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "mean_baseline_mae": float(mean_absolute_error(y_test, baseline)),
    }
    joblib.dump(model, ARTIFACTS / "wine_quality_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
