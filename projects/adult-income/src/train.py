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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
PROCESSED = DATA / "processed"
ARTIFACTS = PROJECT / "artifacts"
URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def download_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    request = Request(URL, headers={"User-Agent": "ml-portfolio/1.0"})
    with urlopen(request, timeout=60) as response:
        archive = ZipFile(BytesIO(response.read()))
    train_name = next(name for name in archive.namelist() if name.endswith("adult.data"))
    test_name = next(name for name in archive.namelist() if name.endswith("adult.test"))
    with archive.open(train_name) as stream:
        train = pd.read_csv(stream, names=COLUMNS, skipinitialspace=True, na_values="?", comment="|", skip_blank_lines=True)
    with archive.open(test_name) as stream:
        test = pd.read_csv(stream, names=COLUMNS, skipinitialspace=True, na_values="?", comment="|", skip_blank_lines=True, skiprows=1)
    return train, test


def clean_split(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [column.strip().lower() for column in cleaned.columns]
    text_columns = [column for column in cleaned.columns if pd.api.types.is_string_dtype(cleaned[column])]
    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()
    cleaned["income"] = cleaned["income"].str.rstrip(".").str.strip()
    cleaned = cleaned.dropna(subset=["income"]).reset_index(drop=True)
    return cleaned


def main() -> None:
    train_raw, test_raw = download_splits()
    train = clean_split(train_raw)
    test = clean_split(test_raw)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "adult_train_clean.csv", index=False)
    test.to_csv(PROCESSED / "adult_test_clean.csv", index=False)

    features = [column for column in train.columns if column != "income"]
    categorical = [column for column in features if column not in NUMERIC]
    X_train, y_train_raw = train[features], train["income"]
    X_test, y_test_raw = test[features], test["income"]
    label_map = {"<=50K": 0, ">50K": 1}
    y_train = y_train_raw.map(label_map)
    y_test = y_test_raw.map(label_map)
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unexpected income labels in the downloaded data.")

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
        "train_rows_raw": int(len(train_raw)),
        "train_rows_used": int(len(train)),
        "test_rows_raw": int(len(test_raw)),
        "test_rows_used": int(len(test)),
        "train_missing_cells_before_imputation": int(train[features].isna().sum().sum()),
        "test_missing_cells_before_imputation": int(test[features].isna().sum().sum()),
        "features_used": features,
        "split": "official UCI adult.data / adult.test split",
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    joblib.dump(model, ARTIFACTS / "adult_income_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
