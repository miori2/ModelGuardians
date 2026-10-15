"""Monitoring session storage and statistical signal calculation."""

from __future__ import annotations

import queue
import statistics
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MonitoringSession:
    model_name: str
    baseline: dict[str, float] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: list[queue.Queue[dict[str, Any]]] = field(default_factory=list)

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

    def create(self, model_name: str, baseline: dict[str, float]) -> MonitoringSession:
        session = MonitoringSession(model_name=model_name, baseline=baseline)
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

    def add_event(self, session: MonitoringSession, event: dict[str, Any]) -> None:
        with self._lock:
            session.events.append(event)
            for subscriber in list(session.subscribers):
                subscriber.put(event)

    def subscribe(self, session: MonitoringSession) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            session.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(
        self, session: MonitoringSession, subscriber: queue.Queue[dict[str, Any]]
    ) -> None:
        with self._lock:
            if subscriber in session.subscribers:
                session.subscribers.remove(subscriber)


def numeric_values(event: dict[str, Any]) -> list[float]:
    values = event.get("features", event.get("prediction"))
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return [float(values)]
    if isinstance(values, list):
        return [float(value) for value in values if isinstance(value, (int, float))]
    return []


def evaluate_event(
    event: dict[str, Any], baseline: dict[str, float]
) -> dict[str, Any]:
    values = numeric_values(event)
    mean = statistics.fmean(values) if values else None
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    baseline_mean = baseline.get("mean")
    baseline_stddev = max(baseline.get("stddev", 0.0), 1e-9)
    z_score = (
        abs(mean - baseline_mean) / baseline_stddev
        if mean is not None and baseline_mean is not None
        else None
    )
    anomaly = z_score is not None and z_score >= 3
    drift = (
        z_score is not None and z_score >= 2
    ) or (baseline_mean is not None and mean is not None and stddev > baseline_stddev * 2)
    return {
        "anomaly": anomaly,
        "drift_indicator": drift,
        "z_score": round(z_score, 4) if z_score is not None else None,
        "observed_mean": mean,
        "observed_stddev": stddev,
    }
