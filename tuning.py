"""In-memory hyperparameter tuning jobs and candidate validation."""

from __future__ import annotations

import json
import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


def _finite_number(value: Any) -> bool:
    return type(value) is int or (
        isinstance(value, float) and math.isfinite(value)
    )


def _serialized(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalize_search_space(search_space: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty object")
    if len(search_space) > 20:
        raise ValueError("search_space must not contain more than 20 parameters")

    normalized: dict[str, dict[str, Any]] = {}
    for name, domain in search_space.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise ValueError("search_space parameter names must be non-empty strings")
        name = name.strip()
        if name in normalized:
            raise ValueError(f"duplicate search_space parameter: {name}")

        if isinstance(domain, list):
            values = domain
            domain_type = "categorical"
        elif isinstance(domain, dict) and domain.get("type") == "categorical":
            if set(domain) != {"type", "values"}:
                raise ValueError(f"{name} categorical domain requires only type and values")
            values = domain.get("values")
            domain_type = "categorical"
        elif (
            isinstance(domain, dict)
            and isinstance(domain.get("type"), str)
            and domain.get("type") in {"int", "float"}
        ):
            if set(domain) != {"type", "min", "max"}:
                raise ValueError(f"{name} numeric domain requires type, min, and max")
            domain_type = domain["type"]
            lower, upper = domain["min"], domain["max"]
            if not _finite_number(lower) or not _finite_number(upper):
                raise ValueError(f"{name} bounds must be finite numbers")
            if domain_type == "int":
                if type(lower) is not int or type(upper) is not int:
                    raise ValueError(f"{name} integer bounds must be integers")
            else:
                try:
                    lower, upper = float(lower), float(upper)
                except OverflowError as exc:
                    raise ValueError(f"{name} bounds must be finite numbers") from exc
                if not math.isfinite(lower) or not math.isfinite(upper):
                    raise ValueError(f"{name} bounds must be finite numbers")
            if lower > upper:
                raise ValueError(f"{name} minimum must not exceed maximum")
            normalized[name] = {"type": domain_type, "min": lower, "max": upper}
            continue
        else:
            raise ValueError(
                f"{name} must be a list of values or a typed categorical/numeric domain"
            )

        if not isinstance(values, list) or not values or len(values) > 100:
            raise ValueError(f"{name} values must be a non-empty list of at most 100 items")
        if any(
            not (
                isinstance(value, (str, bool))
                or type(value) is int
                or (isinstance(value, float) and math.isfinite(value))
            )
            for value in values
        ):
            raise ValueError(f"{name} values must be finite JSON scalar values")
        if len({_serialized(value) for value in values}) != len(values):
            raise ValueError(f"{name} values must not contain duplicates")
        normalized[name] = {"type": domain_type, "values": values}
    return normalized


def _candidate_is_valid(
    candidate: Any, search_space: dict[str, dict[str, Any]]
) -> bool:
    if not isinstance(candidate, dict) or set(candidate) != set(search_space):
        return False
    for name, domain in search_space.items():
        value = candidate[name]
        if domain["type"] == "categorical":
            if not (
                isinstance(value, (str, bool))
                or type(value) is int
                or (isinstance(value, float) and math.isfinite(value))
            ):
                return False
            if _serialized(value) not in {
                _serialized(item) for item in domain["values"]
            }:
                return False
        elif domain["type"] == "int":
            if type(value) is not int or not domain["min"] <= value <= domain["max"]:
                return False
        elif not _finite_number(value) or not domain["min"] <= value <= domain["max"]:
            return False
    return True


def _fallback_candidates(
    search_space: dict[str, dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(count):
        candidate: dict[str, Any] = {}
        categorical_multiplier = 1
        for name, domain in search_space.items():
            if domain["type"] == "categorical":
                values = domain["values"]
                candidate[name] = values[
                    (index // categorical_multiplier) % len(values)
                ]
                categorical_multiplier *= len(values)
            elif domain["type"] == "int":
                candidate[name] = domain["min"] + (
                    (domain["max"] - domain["min"]) * index // max(count - 1, 1)
                )
            else:
                fraction = index / max(count - 1, 1)
                candidate[name] = (
                    domain["min"] * (1 - fraction) + domain["max"] * fraction
                )
        key = _serialized(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def validated_candidates(
    proposed: Any,
    search_space: dict[str, dict[str, Any]],
    count: int,
) -> list[tuple[dict[str, Any], str]]:
    candidates: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    if isinstance(proposed, list):
        for candidate in proposed:
            if len(candidates) >= count:
                break
            if not _candidate_is_valid(candidate, search_space):
                continue
            key = _serialized(candidate)
            if key not in seen:
                candidates.append(
                    ({name: candidate[name] for name in search_space}, "ollama")
                )
                seen.add(key)

    for candidate in _fallback_candidates(search_space, count):
        if len(candidates) >= count:
            break
        key = _serialized(candidate)
        if key not in seen:
            candidates.append((candidate, "fallback"))
            seen.add(key)
    return candidates


@dataclass
class TuningJob:
    model_name: str
    search_space: dict[str, dict[str, Any]]
    metric_name: str
    direction: str
    candidates: list[dict[str, Any]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trials: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        best_trial = self.best_trial()
        submitted = {trial["candidate_id"] for trial in self.trials}
        status = (
            "completed"
            if len(submitted) == len(self.candidates)
            else "running"
            if self.trials
            else "pending"
        )
        return {
            "id": self.id,
            "model_name": self.model_name,
            "search_space": self.search_space,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "status": status,
            "max_trials": len(self.candidates),
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "config": candidate["config"],
                    "source": candidate["source"],
                    "submitted": candidate["candidate_id"] in submitted,
                }
                for candidate in self.candidates
            ],
            "trials": self.trials,
            "best_trial": best_trial,
        }

    def best_trial(self) -> dict[str, Any] | None:
        if not self.trials:
            return None
        return (
            max if self.direction == "maximize" else min
        )(self.trials, key=lambda trial: trial["metric_value"])


class TuningStore:
    """Thread-safe in-memory store for tuning jobs and submitted trials."""

    def __init__(self) -> None:
        self._jobs: dict[str, TuningJob] = {}
        self._lock = threading.RLock()

    def create(
        self,
        model_name: str,
        search_space: Any,
        metric_name: str,
        direction: str,
        max_trials: int,
        proposed: Any = None,
    ) -> TuningJob:
        normalized_space = normalize_search_space(search_space)
        candidates = validated_candidates(proposed, normalized_space, max_trials)
        if not candidates:
            raise ValueError("search_space does not provide any unique candidate configs")
        job = TuningJob(
            model_name=model_name,
            search_space=normalized_space,
            metric_name=metric_name,
            direction=direction,
            candidates=[
                {
                    "candidate_id": f"candidate-{index + 1}",
                    "config": config,
                    "source": source,
                }
                for index, (config, source) in enumerate(candidates)
            ],
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> TuningJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def public_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.public() for job in self._jobs.values()]

    def submit_trial(
        self, job: TuningJob, candidate_id: str, metric_value: int | float
    ) -> dict[str, Any]:
        with self._lock:
            candidate = next(
                (
                    candidate
                    for candidate in job.candidates
                    if candidate["candidate_id"] == candidate_id
                ),
                None,
            )
            if candidate is None:
                raise KeyError("candidate not found")
            if any(trial["candidate_id"] == candidate_id for trial in job.trials):
                raise ValueError("a trial has already been submitted for this candidate")
            trial = {
                "id": str(uuid.uuid4()),
                "candidate_id": candidate_id,
                "config": candidate["config"],
                "metric_name": job.metric_name,
                "metric_value": metric_value,
            }
            job.trials.append(trial)
            return trial
