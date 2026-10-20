
"""Deterministic alert policy."""

from __future__ import annotations

import uuid
from typing import Any


def evaluate_alert(
    signals: dict[str, Any],
    prediction: dict[str, Any] | None = None,
    model_name: str = "unknown",
    model_version: str | None = None,
) -> dict[str, Any]:
    prediction = prediction or {}
    metric_predictions = prediction.get("metrics", {})

    predicted_breach = any(
        isinstance(item, dict)
        and item.get("predicted_breach") is True
        for item in metric_predictions.values()
    )

    if signals.get("anomaly") is True:
        severity = "critical"
        alert_type = "statistical_anomaly"
        action = "investigate"

    elif predicted_breach:
        severity = "warning"
        alert_type = "predictive_warning"
        action = "validate_model"

    elif signals.get("drift_indicator") is True:
        severity = "warning"
        alert_type = "drift"
        action = "investigate"

    else:
        severity = "info"
        alert_type = "normal"
        action = "none"

    return {
        "alert_id": str(uuid.uuid4()),
        "model_name": model_name,
        "model_version": model_version,
        "severity": severity,
        "alert_type": alert_type,
        "status": (
            "open" if severity != "info" else "closed"
        ),
        "recommended_action": action,
        "action_status": (
            "pending" if severity != "info" else "not_required"
        ),
    }
