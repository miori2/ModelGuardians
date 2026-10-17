
"""Simple short-horizon forecasts for numeric monitoring metrics."""

from __future__ import annotations

import math
from typing import Any


def forecast_metrics(
    history: dict[str, list[float]],
    thresholds: dict[str, Any] | None = None,
    horizon: int = 5,
    min_points: int = 4,
) -> dict[str, Any]:
    """Forecast metrics with a least-squares line over event index."""

    thresholds = thresholds or {}
    result: dict[str, Any] = {}
    horizon = max(1, min(int(horizon), 20))

    for name, raw_values in history.items():
        values = [
            float(value)
            for value in raw_values
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            )
        ]

        if len(values) < min_points:
            result[name] = {
                "available": False,
                "reason": (
                    f"Need at least {min_points} valid observations."
                ),
                "observations": len(values),
            }
            continue

        x = list(range(len(values)))
        x_mean = sum(x) / len(x)
        y_mean = sum(values) / len(values)

        denominator = sum(
            (xi - x_mean) ** 2 for xi in x
        )

        slope = (
            sum(
                (xi - x_mean) * (yi - y_mean)
                for xi, yi in zip(x, values)
            ) / denominator
            if denominator
            else 0.0
        )

        predictions = [
            y_mean + slope * (len(values) + offset - x_mean)
            for offset in range(horizon)
        ]

        threshold = thresholds.get(name, {})
        max_threshold = (
            threshold.get("max")
            if isinstance(threshold, dict)
            else None
        )
        min_threshold = (
            threshold.get("min")
            if isinstance(threshold, dict)
            else None
        )

        breach_step = None

        if (
            isinstance(max_threshold, (int, float))
            and not isinstance(max_threshold, bool)
        ):
            breach_step = next(
                (
                    i + 1
                    for i, value in enumerate(predictions)
                    if value > max_threshold
                ),
                None,
            )

        elif (
            isinstance(min_threshold, (int, float))
            and not isinstance(min_threshold, bool)
        ):
            breach_step = next(
                (
                    i + 1
                    for i, value in enumerate(predictions)
                    if value < min_threshold
                ),
                None,
            )

        result[name] = {
            "available": True,
            "observations": len(values),
            "last_observed": values[-1],
            "slope_per_event": round(slope, 6),
            "horizon": horizon,
            "forecast": [
                round(value, 6) for value in predictions
            ],
            "threshold_max": max_threshold,
            "threshold_min": min_threshold,
            "predicted_breach": breach_step is not None,
            "predicted_breach_step": breach_step,
        }

    return result
