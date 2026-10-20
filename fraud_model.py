"""Dataset loading, training, evaluation, and inference for fraud detection."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATASET_HANDLE = "nelgiriyewithana/credit-card-fraud-detection-dataset-2023"
TARGET_COLUMN = "fraud"
IDENTIFIER_COLUMNS = {"id"}
DEFAULT_DATA_PATH = Path("data/raw/credit_card_fraud_2023.csv")
DEFAULT_MODEL_PATH = Path("artifacts/fraud_detector.joblib")
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "max_iter": 100,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
}


def download_dataset(
    destination: Path = DEFAULT_DATA_PATH,
    download_dir: Path = Path("data/kagglehub"),
) -> Path:
    """Download the public Kaggle dataset and copy its CSV into the project data folder."""
    if destination.is_file():
        return destination

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the dataset; install requirements.txt"
        ) from exc

    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(download_dir),
        )
    )
    csv_files = sorted(downloaded.rglob("*.csv")) if downloaded.is_dir() else []
    if downloaded.is_file() and downloaded.suffix.lower() == ".csv":
        csv_files = [downloaded]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file was found in Kaggle download directory: {downloaded}"
        )
    if len(csv_files) > 1:
        named = [
            path
            for path in csv_files
            if "fraud" in path.name.casefold() and path.stat().st_size > 0
        ]
        if len(named) == 1:
            csv_files = named
        else:
            raise ValueError(
                "Expected one fraud dataset CSV in the Kaggle download, found: "
                + ", ".join(str(path) for path in csv_files)
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if csv_files[0].resolve() != destination.resolve():
        shutil.copy2(csv_files[0], destination)
    return destination


def load_dataset(
    csv_path: Path | str,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load and validate the fraud dataset; return features and target column name."""
    frame = pd.read_csv(csv_path)
    target_matches = [
        column
        for column in frame.columns
        if str(column).strip().casefold() in {TARGET_COLUMN, "class"}
    ]
    if len(target_matches) != 1:
        raise ValueError(
            "Expected exactly one fraud target column named 'fraud' or 'Class'; "
            f"found {target_matches or 'none'}"
        )

    target_column = target_matches[0]
    frame = frame.dropna(subset=[target_column]).copy()
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Fraud labels must contain only binary values 0 and 1")
    frame[target_column] = target.astype("int8")
    excluded_columns = [
        column
        for column in frame.columns
        if column != target_column and str(column).strip().casefold() in IDENTIFIER_COLUMNS
    ]
    features = frame.drop(columns=[target_column, *excluded_columns])
    if features.empty:
        raise ValueError("The dataset must contain at least one feature column")
    for column in features.columns:
        try:
            features[column] = pd.to_numeric(features[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature column {column!r} must be numeric") from exc
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.isna().all(axis=1).any():
        raise ValueError("Each dataset row must contain at least one numeric feature")
    return features, frame[target_column], target_column


def _estimator(params: dict[str, Any], random_state: int):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=random_state,
            **params,
        ),
    )


@dataclass
class FraudModel:
    estimator: Any
    feature_columns: list[str]
    target_column: str
    baseline: dict[str, float]
    metrics: dict[str, Any]
    dataset_summary: dict[str, Any]
    model_params: dict[str, Any]

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(features, dict):
            raise ValueError("features must be an object keyed by feature name")
        missing = [name for name in self.feature_columns if name not in features]
        if missing:
            raise ValueError("Missing required feature(s): " + ", ".join(missing))

        values = []
        for name in self.feature_columns:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Feature {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"Feature {name!r} must be finite")
            values.append(float(value))

        row = pd.DataFrame([values], columns=self.feature_columns)
        probabilities = self.estimator.predict_proba(row)[0]
        classes = list(self.estimator[-1].classes_)
        fraud_class_index = classes.index(1)
        probability = float(probabilities[fraud_class_index])
        predicted_class = int(self.estimator.predict(row)[0])
        return {
            "fraud_probability": probability,
            "predicted_class": predicted_class,
        }

    def public(self) -> dict[str, Any]:
        return {
            "model_name": "credit-card-fraud-detector",
            "target_column": self.target_column,
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "baseline": self.baseline,
            "metrics": self.metrics,
            "dataset_summary": self.dataset_summary,
            "model_params": self.model_params,
        }


def train_fraud_model(
    csv_path: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, Any] | None = None,
) -> FraudModel:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    features, target, target_column = load_dataset(csv_path)
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    if target.nunique() != 2:
        raise ValueError("Training data must contain both fraud labels 0 and 1")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    estimator = _estimator(params, random_state)
    estimator.fit(x_train, y_train)

    positive_class_index = list(estimator[-1].classes_).index(1)
    test_probabilities = estimator.predict_proba(x_test)[:, positive_class_index]
    predictions = estimator.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "average_precision": float(average_precision_score(y_test, test_probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "test_rows": int(len(y_test)),
    }

    train_probabilities = estimator.predict_proba(x_train)[:, positive_class_index]
    baseline = {
        "mean": float(np.mean(train_probabilities)),
        "stddev": float(np.std(train_probabilities)),
    }
    fraud_count = int(target.sum())
    bundle = FraudModel(
        estimator=estimator,
        feature_columns=list(features.columns),
        target_column=target_column,
        baseline=baseline,
        metrics=metrics,
        dataset_summary={
            "row_count": int(len(features)),
            "fraud_count": fraud_count,
            "legitimate_count": int(len(features) - fraud_count),
            "fraud_rate": float(fraud_count / len(features)),
        },
        model_params=params,
    )
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def score_candidate(
    features: pd.DataFrame,
    target: pd.Series,
    model_params: dict[str, Any],
    *,
    random_state: int = 42,
) -> float:
    """Score a tuning candidate on a stratified validation split."""
    x_train, x_validation, y_train, y_validation = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    estimator = _estimator(model_params, random_state)
    estimator.fit(x_train, y_train)
    positive_class_index = list(estimator[-1].classes_).index(1)
    probabilities = estimator.predict_proba(x_validation)[:, positive_class_index]
    return float(average_precision_score(y_validation, probabilities))


def load_fraud_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> FraudModel:
    model = joblib.load(model_path)
    if not isinstance(model, FraudModel):
        raise TypeError(f"Expected a FraudModel artifact at {model_path}")
    return model
