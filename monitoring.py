
"""Metric-level monitoring, baselines, and in-memory event storage."""

from __future__ import annotations

import math
import queue
import statistics
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass
class MonitoringSession:
    model_name: str
    baseline: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    metric_history: dict[str, list[float]] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "baseline": self.baseline,
            "event_count": len(self.events),
        }


class MonitoringStore:
    """Thread-safe in-memory store for active monitoring sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, MonitoringSession] = {}
        self._lock = threading.RLock()

    def create(
        self, model_name: str, baseline: dict[str, Any]
    ) -> MonitoringSession:
        session = MonitoringSession(
            model_name=model_name,
            baseline=baseline,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> MonitoringSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    **session.public(),
                    "events": [
                        {
                            **event,
                            "signals": dict(event.get("signals", {})),
                        }
                        for event in session.events
                    ],
                }
                for session in self._sessions.values()
            ]

    def add_event(
        self,
        session: MonitoringSession,
        event: dict[str, Any],
    ) -> None:
        with self._lock:
            session.events.append(event)

            for metric, value in event.get("metrics", {}).items():
                if _finite_number(value):
                    history = session.metric_history.setdefault(
                        metric, []
                    )
                    history.append(float(value))

                    # Keep at most 500 observations per metric.
                    del history[:-500]

            for subscriber in list(session.subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    # A slow dashboard client must not block monitoring.
                    pass

    def subscribe(self, session: MonitoringSession) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=100)

        with self._lock:
            session.subscribers.append(subscriber)

        return subscriber

    def unsubscribe(
        self,
        session: MonitoringSession,
        subscriber: queue.Queue,
    ) -> None:
        with self._lock:
            if subscriber in session.subscribers:
                session.subscribers.remove(subscriber)


def numeric_values(event: dict[str, Any]) -> list[float]:
    """Legacy helper retained for compatibility with earlier code."""

    prediction = event.get("prediction")
    values = prediction if _finite_number(prediction) else event.get("features")
    if _finite_number(values):
        return [float(values)]

    if isinstance(values, list):
        return [
            float(value)
            for value in values
            if _finite_number(value)
        ]

    return []


def _baseline_for_metric(
    baseline: dict[str, Any],
    metric: str,
) -> dict[str, float] | None:
    """Support both named metric baselines and the older mean/stddev format."""

    metrics = baseline.get("metrics")

    if isinstance(metrics, dict) and isinstance(
        metrics.get(metric), dict
    ):
        item = metrics[metric]
        mean = item.get("mean")
        stddev = item.get("stddev", 0.0)

        if (
            _finite_number(mean)
            and _finite_number(stddev)
            and float(stddev) >= 0
        ):
            return {
                "mean": float(mean),
                "stddev": float(stddev),
            }

    if (
        _finite_number(baseline.get("mean"))
        and _finite_number(baseline.get("stddev", 0.0))
    ):
        return {
            "mean": float(baseline["mean"]),
            "stddev": max(
                0.0,
                float(baseline.get("stddev", 0.0)),
            ),
        }

    return None


def evaluate_event(
    event: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare each named metric against its baseline."""

    metrics = event.get("metrics")

    # Support older events that contain features or prediction instead.
    if not isinstance(metrics, dict) or not metrics:
        values = numeric_values(event)
        mean = statistics.fmean(values) if values else None
        stddev = (
            statistics.pstdev(values)
            if len(values) > 1
            else 0.0
        )

        base = _baseline_for_metric(baseline, "value")

        if base is None and _finite_number(baseline.get("mean")):
            base = {
                "mean": float(baseline["mean"]),
                "stddev": float(baseline.get("stddev", 0.0)),
            }

        z_score = None

        if mean is not None and base is not None:
            if base["stddev"] == 0:
                z_score = (
                    0.0 if mean == base["mean"] else None
                )
            else:
                z_score = (
                    abs(mean - base["mean"]) / base["stddev"]
                )

        return {
            "anomaly": z_score is not None and z_score >= 3,
            "drift_indicator": z_score is not None and z_score >= 2,
            "z_score": (
                round(z_score, 4)
                if z_score is not None
                else None
            ),
            "observed_mean": mean,
            "observed_stddev": stddev,
            "metrics": {},
            "notes": (
                "Legacy aggregate evaluation; "
                "prefer named event.metrics."
            ),
        }

    per_metric: dict[str, Any] = {}
    any_anomaly = False
    any_drift = False
    valid_values: list[float] = []

    for name, value in metrics.items():
        if not isinstance(name, str) or not _finite_number(value):
            per_metric[str(name)] = {
                "valid": False,
                "reason": "Metric must be a finite number.",
            }
            continue

        observed = float(value)
        valid_values.append(observed)
        base = _baseline_for_metric(baseline, name)

        if base is None:
            per_metric[name] = {
                "valid": True,
                "observed_value": observed,
                "baseline_available": False,
                "anomaly": False,
                "drift_indicator": False,
                "z_score": None,
            }
            continue

        delta = abs(observed - base["mean"])

        if base["stddev"] == 0:
            tolerance = max(abs(base["mean"]) * 0.01, 1e-9)
            z_score = None
            anomaly = delta > tolerance
            drift = delta > tolerance
        else:
            z_score = delta / base["stddev"]
            anomaly = z_score >= 3
            drift = z_score >= 2

        any_anomaly = any_anomaly or anomaly
        any_drift = any_drift or drift

        per_metric[name] = {
            "valid": True,
            "observed_value": observed,
            "baseline_mean": base["mean"],
            "baseline_stddev": base["stddev"],
            "absolute_deviation": round(delta, 6),
            "z_score": (
                round(z_score, 4)
                if z_score is not None
                else None
            ),
            "anomaly": anomaly,
            "drift_indicator": drift,
        }

    return {
        "anomaly": any_anomaly,
        "drift_indicator": any_drift,
        "z_score": max(
            (
                item["z_score"]
                for item in per_metric.values()
                if isinstance(item, dict)
                and isinstance(
                    item.get("z_score"), (int, float)
                )
            ),
            default=None,
        ),
        "observed_mean": (
            statistics.fmean(valid_values)
            if valid_values
            else None
        ),
        "observed_stddev": (
            statistics.pstdev(valid_values)
            if len(valid_values) > 1
            else 0.0
        ),
        "metrics": per_metric,
    }
