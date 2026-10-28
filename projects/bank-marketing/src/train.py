from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
DATA = PROJECT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = PROJECT / "artifacts"
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET = "y"
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]


def download_frame() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "ml-portfolio/1.0"})
    with urlopen(request, timeout=60) as response:
        outer_archive = ZipFile(BytesIO(response.read()))
    nested_name = next(name for name in outer_archive.namelist() if name.endswith("bank.zip"))
    archive = ZipFile(BytesIO(outer_archive.read(nested_name)))
    member = next(name for name in archive.namelist() if name.endswith("bank-full.csv"))
    with archive.open(member) as stream:
        return pd.read_csv(stream, sep=";")


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]
    text_columns = [column for column in cleaned.columns if pd.api.types.is_string_dtype(cleaned[column])]
    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()
    cleaned[TARGET] = cleaned[TARGET].str.lower()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    categorical = [column for column in cleaned.columns if column not in NUMERIC + [TARGET]]
    cleaned[categorical] = cleaned[categorical].replace("unknown", np.nan)
    return cleaned


def main() -> None:
    frame = download_frame()
    cleaned = clean_data(frame)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED / "bank_marketing_clean.csv", index=False)

    features = [column for column in cleaned.columns if column != TARGET and column != "duration"]
    categorical = [column for column in features if column not in NUMERIC]
    X = cleaned[features]
    y = cleaned[TARGET].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected target labels; expected only 'yes' and 'no'.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "task": "binary classification",
        "rows_raw": int(len(frame)),
        "rows_after_deduplication": int(len(cleaned)),
        "positive_rate": float(y.mean()),
        "features_used": features,
        "excluded_features": {"duration": "Only known after a call; excluded to avoid target leakage."},
        "test_size": 0.2,
        "random_state": 42,
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    joblib.dump(model, ARTIFACTS / "bank_marketing_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
