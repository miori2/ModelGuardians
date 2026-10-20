
"""In-memory hyperparameter tuning jobs and candidate validation."""

from __future__ import annotations

import itertools
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
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalize_search_space(
    search_space: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty object")

    if len(search_space) > 20:
        raise ValueError(
            "search_space must not contain more than 20 parameters"
        )

    normalized = {}

    for name, domain in search_space.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name) > 100
        ):
            raise ValueError(
                "parameter names must be non-empty strings"
            )

        name = name.strip()

        if name in normalized:
            raise ValueError(f"duplicate parameter: {name}")

        if isinstance(domain, list):
            values, kind = domain, "categorical"

        elif (
            isinstance(domain, dict)
            and domain.get("type") == "categorical"
            and set(domain) == {"type", "values"}
        ):
            values, kind = domain["values"], "categorical"

        elif (
            isinstance(domain, dict)
            and domain.get("type") in {"int", "float"}
            and set(domain) == {"type", "min", "max"}
        ):
            kind = domain["type"]
            lo, hi = domain["min"], domain["max"]

            if (
                not _finite_number(lo)
                or not _finite_number(hi)
            ):
                raise ValueError(
                    f"{name} bounds must be finite numbers"
                )

            if kind == "int" and (
                type(lo) is not int or type(hi) is not int
            ):
                raise ValueError(
                    f"{name} integer bounds must be integers"
                )

            if kind == "float":
                lo, hi = float(lo), float(hi)

                if not math.isfinite(lo) or not math.isfinite(hi):
                    raise ValueError(
                        f"{name} bounds must be finite"
                    )

            if lo > hi:
                raise ValueError(
                    f"{name} min must not exceed max"
                )

            normalized[name] = {
                "type": kind,
                "min": lo,
                "max": hi,
            }
            continue

        else:
            raise ValueError(
                f"{name} must be a categorical list "
                "or typed numeric domain"
            )

        if (
            not isinstance(values, list)
            or not values
            or len(values) > 100
        ):
            raise ValueError(
                f"{name} values must be a non-empty list "
                "of at most 100"
            )

        if any(
            not (
                isinstance(value, (str, bool))
                or type(value) is int
                or (
                    isinstance(value, float)
                    and math.isfinite(value)
                )
            )
            for value in values
        ):
            raise ValueError(
                f"{name} values must be finite JSON scalars"
            )

        if len({_serialized(value) for value in values}) != len(values):
            raise ValueError(
                f"{name} values must not contain duplicates"
            )

        normalized[name] = {
            "type": kind,
            "values": values,
        }

    return normalized


def _valid(
    candidate: Any,
    space: dict[str, dict[str, Any]],
) -> bool:
    if not isinstance(candidate, dict) or set(candidate) != set(space):
        return False

    for name, domain in space.items():
        value = candidate[name]

        if domain["type"] == "categorical":
            allowed = {
                _serialized(item)
                for item in domain["values"]
            }

            if _serialized(value) not in allowed:
                return False

        elif domain["type"] == "int":
            if (
                type(value) is not int
                or not domain["min"] <= value <= domain["max"]
            ):
                return False

        elif (
            not _finite_number(value)
            or not domain["min"] <= value <= domain["max"]
        ):
            return False

    return True


def _fallback(
    space: dict[str, dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    result = []
    seen = set()

    for index in range(count):
        item = {}
        multiplier = 1

        for name, domain in space.items():
            if domain["type"] == "categorical":
                values = domain["values"]
                item[name] = values[
                    (index // multiplier) % len(values)
                ]
                multiplier *= len(values)

            elif domain["type"] == "int":
                item[name] = domain["min"] + (
                    (domain["max"] - domain["min"])
                    * index
                    // max(count - 1, 1)
                )

            else:
                fraction = index / max(count - 1, 1)
                item[name] = (
                    domain["min"] * (1 - fraction)
                    + domain["max"] * fraction
                )

        key = _serialized(item)

        if key not in seen:
            result.append(item)
            seen.add(key)

    return result


def validated_candidates(
    proposed: Any,
    space: dict[str, dict[str, Any]],
    count: int,
) -> list[tuple[dict[str, Any], str]]:
    result = []
    seen = set()

    if isinstance(proposed, list):
        for item in proposed:
            if len(result) >= count:
                break

            if _valid(item, space):
                key = _serialized(item)

                if key not in seen:
                    result.append(
                        (
                            {name: item[name] for name in space},
                            "ollama",
                        )
                    )
                    seen.add(key)

    for item in _fallback(space, count):
        if len(result) >= count:
            break

        key = _serialized(item)

        if key not in seen:
            result.append((item, "fallback"))
            seen.add(key)

    return result


@dataclass
class TuningJob:
    model_name: str
    search_space: dict[str, dict[str, Any]]
    metric_name: str
    direction: str
    candidates: list[dict[str, Any]]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trials: list[dict[str, Any]] = field(default_factory=list)

    def best_trial(self):
        if not self.trials:
            return None

        chooser = (
            max if self.direction == "maximize" else min
        )

        return chooser(
            self.trials,
            key=lambda trial: trial["metric_value"],
        )

    def public(self):
        submitted = {
            trial["candidate_id"]
            for trial in self.trials
        }

        return {
            "id": self.id,
            "model_name": self.model_name,
            "search_space": self.search_space,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "status": (
                "completed"
                if len(submitted) == len(self.candidates)
                else (
                    "running"
                    if self.trials
                    else "pending"
                )
            ),
            "max_trials": len(self.candidates),
            "candidates": [
                {
                    **candidate,
                    "submitted": (
                        candidate["candidate_id"] in submitted
                    ),
                }
                for candidate in self.candidates
            ],
            "trials": self.trials,
            "best_trial": self.best_trial(),
        }


class TuningStore:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.RLock()

    def create(
        self,
        model_name,
        search_space,
        metric_name,
        direction,
        max_trials,
        proposed=None,
    ):
        space = normalize_search_space(search_space)

        candidates = validated_candidates(
            proposed,
            space,
            max_trials,
        )

        if len(candidates) < max_trials:
            raise ValueError(
                f"search_space has only {len(candidates)} unique candidate configs"
            )

        job = TuningJob(
            model_name,
            space,
            metric_name,
            direction,
            [
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

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def public_jobs(self):
        with self._lock:
            return [
                job.public()
                for job in self._jobs.values()
            ]

    def submit_trial(
        self,
        job,
        candidate_id,
        metric_value,
    ):
        with self._lock:
            candidate = next(
                (
                    item
                    for item in job.candidates
                    if item["candidate_id"] == candidate_id
                ),
                None,
            )

            if candidate is None:
                raise KeyError("candidate not found")

            if any(
                trial["candidate_id"] == candidate_id
                for trial in job.trials
            ):
                raise ValueError(
                    "a trial has already been submitted for this candidate"
                )

            trial = {
                "id": str(uuid.uuid4()),
                "candidate_id": candidate_id,
                "config": candidate["config"],
                "metric_name": job.metric_name,
                "metric_value": metric_value,
            }

            job.trials.append(trial)

            return trial
