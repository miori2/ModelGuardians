"""Command-line workflow for training and monitoring the Kaggle fraud dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fraud_model import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    download_dataset,
    load_dataset,
    load_fraud_model,
    score_candidate,
    train_fraud_model,
)


def dataset_path(value: str | None) -> Path:
    if value:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset CSV does not exist: {path}")
        return path
    return download_dataset(DEFAULT_DATA_PATH)


def train_command(args: argparse.Namespace) -> None:
    path = dataset_path(args.data)
    model = train_fraud_model(path, args.model_path)
    print(
        json.dumps(
            {
                "dataset": str(path),
                "model_path": str(args.model_path),
                **model.public(),
            },
            indent=2,
        )
    )


def _rows_for_prediction(
    features: pd.DataFrame, target: pd.Series, limit: int
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    sample = features.join(target.rename("_actual")).sample(
        n=min(limit, len(features)), random_state=42
    )
    rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        values = row.drop(labels=["_actual"])
        rows.append(
            {
                "features": {name: float(value) for name, value in values.items()},
                "actual": int(row["_actual"]),
            }
        )
    return rows


def replay_command(args: argparse.Namespace) -> None:
    path = dataset_path(args.data)
    model = load_fraud_model(args.model_path)
    features, target, _ = load_dataset(path)
    rows = _rows_for_prediction(features, target, args.limit)
    base_url = args.api_url.rstrip("/")
    response = requests.post(
        f"{base_url}/monitoring/sessions",
        json={
            "model_name": "credit-card-fraud-detector",
            "baseline": model.baseline,
        },
        timeout=30,
    )
    response.raise_for_status()
    session = response.json()
    session_id = session["id"]

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        response = requests.post(
            f"{base_url}/monitoring/sessions/{session_id}/predict/batch",
            json={"rows": batch},
            timeout=120,
        )
        response.raise_for_status()
        print(f"Submitted {min(start + len(batch), len(rows))}/{len(rows)} predictions")
        if args.delay > 0 and start + len(batch) < len(rows):
            time.sleep(args.delay)

    print(
        json.dumps(
            {
                "session_id": session_id,
                "prediction_count": len(rows),
                "dashboard": f"{base_url}/dashboard",
            },
            indent=2,
        )
    )


def tune_command(args: argparse.Namespace) -> None:
    path = dataset_path(args.data)
    features, target, _ = load_dataset(path)
    base_url = args.api_url.rstrip("/")
    job_response = requests.post(
        f"{base_url}/tuning/jobs",
        json={
            "model_name": "credit-card-fraud-detector",
            "metric_name": "validation_average_precision",
            "direction": "maximize",
            "max_trials": args.max_trials,
            "search_space": {
                "max_iter": [50, 100, 150],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_leaf_nodes": [15, 31, 63],
            },
        },
        timeout=120,
    )
    job_response.raise_for_status()
    job = job_response.json()

    completed_trials = []
    for candidate in job["candidates"]:
        score = score_candidate(
            features,
            target,
            candidate["config"],
        )
        response = requests.post(
            f"{base_url}/tuning/jobs/{job['id']}/trials",
            json={
                "candidate_id": candidate["candidate_id"],
                "metric_value": score,
            },
            timeout=120,
        )
        response.raise_for_status()
        completed_trials.append(
            {
                "candidate_id": candidate["candidate_id"],
                "config": candidate["config"],
                "validation_average_precision": score,
            }
        )

    print(
        json.dumps(
            {
                "job_id": job["id"],
                "completed_trials": completed_trials,
                "best_trial": max(
                    completed_trials,
                    key=lambda trial: trial["validation_average_precision"],
                ),
                "dashboard": f"{base_url}/dashboard",
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="download (if needed), inspect, train, and evaluate the model"
    )
    train_parser.add_argument("--data", help="CSV path; downloads the Kaggle dataset by default")
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.set_defaults(handler=train_command)

    replay_parser = subparsers.add_parser(
        "replay", help="run predictions on dataset rows and send them to monitoring"
    )
    replay_parser.add_argument("--data", help="CSV path; downloads the Kaggle dataset by default")
    replay_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    replay_parser.add_argument("--api-url", default="http://127.0.0.1:5000")
    replay_parser.add_argument("--limit", type=int, default=500)
    replay_parser.add_argument("--batch-size", type=int, default=100)
    replay_parser.add_argument("--delay", type=float, default=0.25)
    replay_parser.set_defaults(handler=replay_command)

    tune_parser = subparsers.add_parser(
        "tune", help="evaluate tuning candidates and publish their metrics to the dashboard"
    )
    tune_parser.add_argument("--data", help="CSV path; downloads the Kaggle dataset by default")
    tune_parser.add_argument("--api-url", default="http://127.0.0.1:5000")
    tune_parser.add_argument("--max-trials", type=int, choices=range(1, 11), default=3)
    tune_parser.set_defaults(handler=tune_command)

    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError, requests.RequestException) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
