
"""Flask application for monitoring, forecasting, Ollama insights, and alerts."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
import plotly.graph_objects as go
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from alerting import evaluate_alert
from forecasting import forecast_metrics
from monitoring import MonitoringStore, evaluate_event
from ollama_client import OllamaAnalyzer
from tuning import TuningStore, normalize_search_space
from fraud_model import FraudModel, load_fraud_model


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _has_predicted_breach(event: dict[str, Any]) -> bool:
    for source in (event.get("prediction"), event.get("forecast")):
        if not isinstance(source, dict):
            continue
        metrics = source.get("metrics")
        if isinstance(metrics, dict) and any(
            isinstance(metric, dict)
            and metric.get("predicted_breach") is True
            for metric in metrics.values()
        ):
            return True
    return False


def dashboard_analytics(
    sessions: list[dict[str, Any]],
    selected_session_id: str | None,
) -> dict[str, Any]:
    if selected_session_id:
        selected = next(
            (s for s in sessions if s["id"] == selected_session_id),
            None,
        )

        if selected is None:
            raise KeyError("monitoring session not found")

        active_sessions = [selected]
    else:
        active_sessions = sessions

    events = [
        {
            **event,
            "session_id": session["id"],
            "model_name": session["model_name"],
        }
        for session in active_sessions
        for event in session["events"]
    ]

    anomalies = sum(
        bool(event.get("signals", {}).get("anomaly"))
        for event in events
    )

    drifts = sum(
        bool(event.get("signals", {}).get("drift_indicator"))
        for event in events
    )

    alerts = [
        event["alert"]
        for event in events
        if isinstance(event.get("alert"), dict)
    ]

    active_alerts = [
        alert
        for alert in alerts
        if alert.get("status") == "open"
        and alert.get("severity") != "info"
    ]

    forecast = _forecast_window(events)

    return {
        "sessions": [
            {
                key: session[key]
                for key in ("id", "model_name", "event_count")
            }
            for session in sessions
        ],
        "selected_session_id": selected_session_id,
        "events": events[-500:],
        "forecast": forecast,
        "monitoring_figure": monitoring_figure(events, forecast),
        "tuning_figure": {"data": [{"type": "bar", "x": [], "y": []}], "layout": {"title": "Tuning jobs"}},
        "summary": {
            "session_count": len(active_sessions),
            "event_count": len(events),
            "anomaly_count": anomalies,
            "drift_count": drifts,
            "active_alert_count": len(active_alerts),
            "predictive_warning_count": sum(
                _has_predicted_breach(event) for event in events
            ),
            "model_count": len(
                {session["model_name"] for session in active_sessions}
            ),
            "tuning_job_count": 0,
        },
        "active_alerts": active_alerts[-100:],
    }


def _forecast_window(events: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    means = [
        (index, event["signals"]["observed_mean"])
        for index, event in enumerate(events)
        if isinstance(event.get("signals"), dict)
        and _finite_number(event["signals"].get("observed_mean"))
    ]
    if len(means) < 2:
        return []

    x_mean = sum(index for index, _ in means) / len(means)
    y_mean = sum(value for _, value in means) / len(means)
    denominator = sum((index - x_mean) ** 2 for index, _ in means)
    slope = (
        sum((index - x_mean) * (value - y_mean) for index, value in means)
        / denominator
        if denominator
        else 0.0
    )
    horizon = min(5, len(means))
    return [
        {
            "step": len(events) + offset,
            "observed_mean": y_mean + slope * (len(events) + offset - x_mean),
        }
        for offset in range(horizon)
    ]


ROLLING_WINDOW = 20


def monitoring_figure(
    events: list[dict[str, Any]], forecast: list[dict[str, float | int]]
) -> dict[str, Any]:
    observed = [
        event
        for event in events
        if isinstance(event.get("signals"), dict)
        and isinstance(event["signals"].get("observed_mean"), (int, float))
        and not isinstance(event["signals"]["observed_mean"], bool)
        and math.isfinite(event["signals"]["observed_mean"])
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[
                event["timestamp"]
                if isinstance(event.get("timestamp"), str) and event["timestamp"]
                else f"Event {index + 1}"
                for index, event in enumerate(observed)
            ],
            y=[event["signals"]["observed_mean"] for event in observed],
            type="scatter",
            mode="markers" if len(observed) > ROLLING_WINDOW else "lines+markers",
            name="Observed mean",
            opacity=0.35 if len(observed) > ROLLING_WINDOW else 1,
            line={"color": "#67a5ff", "width": 2},
            marker={
                "size": 5,
                "color": [
                    "#ff786f" if event["signals"].get("anomaly") else "#67a5ff"
                    for event in observed
                ],
            },
        )
    )
    if forecast and observed:
        figure.add_trace(
            go.Scatter(
                x=[
                    "Forecast start",
                    *[
                        f"Forecast +{index + 1}"
                        for index in range(len(forecast))
                    ],
                ],
                y=[
                    observed[-1]["signals"]["observed_mean"],
                    *[point["observed_mean"] for point in forecast],
                ],
                type="scatter",
                mode="lines+markers",
                name="Linear forecast",
                line={"color": "#bb91ff", "width": 2, "dash": "dash"},
            )
        )
    if len(observed) > 1:
        values = [event["signals"]["observed_mean"] for event in observed]
        rolling = [
            sum(values[max(0, i - ROLLING_WINDOW + 1) : i + 1])
            / len(values[max(0, i - ROLLING_WINDOW + 1) : i + 1])
            for i in range(len(values))
        ]
        figure.add_trace(
            go.Scatter(
                x=[
                    event["timestamp"]
                    if isinstance(event.get("timestamp"), str) and event["timestamp"]
                    else f"Event {index + 1}"
                    for index, event in enumerate(observed)
                ],
                y=rolling,
                type="scatter",
                mode="lines",
                name=f"Rolling average ({ROLLING_WINDOW} events)",
                line={"color": "#ffd166", "width": 3},
            )
        )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a27",
        plot_bgcolor="#111a27",
        font={"color": "#c8d5e9"},
        margin={"l": 48, "r": 18, "t": 12, "b": 48},
        xaxis={"title": "Event sequence", "gridcolor": "#263246"},
        yaxis={"title": "Fraud probability / observed signal", "gridcolor": "#263246"},
        legend={"orientation": "h", "y": -0.22},
    )
    return json.loads(figure.to_json())


def tuning_figure(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    trials = [(job, trial) for job in jobs for trial in job["trials"]]
    figure = go.Figure(
        data=[
            go.Bar(
                x=[
                    f"{job['model_name']} · {job['metric_name']} · {trial['candidate_id']}"
                    for job, trial in trials
                ],
                y=[trial["metric_value"] for _, trial in trials],
                marker={"color": "#43c6a2"},
                hovertemplate="%{x}<br>Metric: %{y}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a27",
        plot_bgcolor="#111a27",
        font={"color": "#c8d5e9"},
        margin={"l": 48, "r": 18, "t": 12, "b": 90},
        xaxis={"tickangle": -25, "gridcolor": "#263246"},
        yaxis={"title": "Trial metric", "gridcolor": "#263246"},
    )
    return json.loads(figure.to_json())


def metric_explanations(
    analytics: dict[str, Any],
    jobs: list[dict[str, Any]],
    fraud_model: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Plain-language explanation of every metric shown on the dashboard."""
    summary = analytics["summary"]
    events = analytics["events"]
    items: list[dict[str, str]] = []

    def add(group: str, metric: str, value: str, explanation: str) -> None:
        items.append(
            {"group": group, "metric": metric, "value": value, "explanation": explanation}
        )

    count = summary["event_count"]
    add(
        "Monitoring",
        "Monitoring sessions",
        str(summary["session_count"]),
        f"{summary['session_count']} session(s) covering {summary['model_count']} model(s). "
        "A session is one stream of predictions being watched.",
    )
    add(
        "Monitoring",
        "Observed events",
        str(count),
        "Predictions recorded so far. More events make the trends and rates below more reliable."
        if count
        else "No predictions have been recorded yet, so nothing can be analyzed.",
    )
    for key, label, meaning in (
        ("anomaly_count", "Anomalies", "events whose signal was 3 or more standard deviations from the baseline"),
        ("drift_count", "Drift signals", "events 2 or more standard deviations from the baseline, or unusually spread out"),
    ):
        value = summary[key]
        rate = f" ({value / count:.1%} of events)" if count else ""
        verdict = (
            "None were flagged, so behavior matches the baseline."
            if value == 0
            else "Review these events; the model's output may be shifting from what it was trained on."
        )
        add("Monitoring", label, str(value), f"{value} {meaning}{rate}. {verdict}")

    probabilities = [
        event["signals"]["observed_mean"]
        for event in events
        if isinstance(event.get("signals"), dict)
        and isinstance(event["signals"].get("observed_mean"), (int, float))
    ]
    if probabilities:
        recent = probabilities[-20:]
        mean_recent = sum(recent) / len(recent)
        add(
            "Observed signal",
            "Recent average signal",
            f"{mean_recent:.3f}",
            f"Average of the last {len(recent)} observed values (fraud probability for the fraud model). "
            "The yellow chart line shows this over time; a sustained move away from the baseline suggests drift.",
        )
    labelled = [e for e in events if e.get("actual") in (0, 1) and "predicted_class" in e]
    if labelled:
        correct = sum(e["predicted_class"] == e["actual"] for e in labelled)
        add(
            "Observed signal",
            "Live accuracy",
            f"{correct / len(labelled):.1%}",
            f"{correct} of {len(labelled)} replayed predictions matched the true label. "
            "This is measured on the events shown, not on the held-out test set.",
        )
    forecast = analytics["forecast"]
    if forecast:
        add(
            "Observed signal",
            "Short-term forecast",
            f"{forecast[0]['observed_mean']:.3f} → {forecast[-1]['observed_mean']:.3f}",
            "A straight-line trend extended over the next few events. It is indicative only; "
            "it is not very informative when values jump between near 0 and near 1.",
        )

    if fraud_model:
        metrics = fraud_model["metrics"]
        descriptions = (
            ("accuracy", "Accuracy", "share of all test predictions that were correct"),
            ("precision", "Precision", "of the transactions flagged as fraud, the share that really were fraud"),
            ("recall", "Recall", "of all real fraud, the share the model caught"),
            ("f1", "F1 score", "balance of precision and recall"),
            ("roc_auc", "ROC-AUC", "how well the model ranks fraud above legitimate cases (1.0 is perfect, 0.5 is chance)"),
            ("average_precision", "Average precision", "precision averaged across all thresholds; the main tuning metric"),
        )
        for key, label, meaning in descriptions:
            if isinstance(metrics.get(key), (int, float)):
                add("Model (held-out test set)", label, f"{metrics[key]:.4f}", f"{meaning[0].upper()}{meaning[1:]}.")
        matrix = metrics.get("confusion_matrix")
        if matrix and len(matrix) == 2:
            (tn, fp), (fn, tp) = matrix
            add(
                "Model (held-out test set)",
                "Confusion matrix",
                f"TN {tn:,} · FP {fp:,} · FN {fn:,} · TP {tp:,}",
                f"{fp:,} legitimate transactions were wrongly flagged (false positives) and {fn:,} frauds were missed "
                "(false negatives). Near-perfect scores on this balanced dataset are optimistic compared with real data.",
            )

    for job in jobs:
        trials = job["trials"]
        best = job["best_trial"]
        if not trials or not best:
            continue
        values = [trial["metric_value"] for trial in trials]
        add(
            "Hyperparameter tuning",
            f"{job['metric_name']} ({job['model_name']})",
            f"best {best['metric_value']:.4f}",
            f"{len(trials)} trial(s) ranged from {min(values):.4f} to {max(values):.4f}; "
            f"{best['candidate_id']} is best ({job['direction']} is better). "
            + (
                "The spread is tiny, so these settings perform almost identically."
                if max(values) - min(values) < 0.01
                else "Settings meaningfully affect results, so further tuning may help."
            ),
        )
    return items


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    analyzer: OllamaAnalyzer | None = None,
    store: MonitoringStore | None = None,
    tuning_store: TuningStore | None = None,
    fraud_detector: FraudModel | None = None,
) -> Flask:
    app = Flask(__name__)

    monitoring_store = store or MonitoringStore()
    hyperparameter_tuning_store = tuning_store or TuningStore()

    llm = analyzer or OllamaAnalyzer(
        host=os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        ),
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        timeout=float(os.getenv("OLLAMA_TIMEOUT", "10")),
    )

    def record_event(
        session: Any,
        event: dict[str, Any],
        *,
        run_diagnosis: bool = True,
        forward_alert: bool = True,
    ) -> dict[str, Any]:
        signals = evaluate_event(event, session.baseline)
        forecast = {
            "metrics": forecast_metrics(
                session.metric_history,
                thresholds=session.baseline.get("_thresholds", {}),
                horizon=5,
                min_points=4,
            )
        }
        version = event.get("model_version")
        alert = evaluate_alert(
            signals,
            forecast,
            model_name=session.model_name,
            model_version=version if isinstance(version, str) else None,
        )
        if run_diagnosis:
            try:
                if hasattr(llm, "assess_alert"):
                    diagnosis = llm.assess_alert(event, signals, forecast)
                else:
                    diagnosis = llm.analyze(event, signals)
            except (requests.RequestException, ValueError, AttributeError) as exc:
                app.logger.warning("Ollama analysis unavailable: %s", exc)
                diagnosis = "Ollama analysis unavailable; review the statistical signals."
        else:
            diagnosis = "LLM diagnosis skipped for batch prediction ingestion."
        enriched = {
            **event,
            "timestamp": event.get("timestamp", utc_now()),
            "model_name": session.model_name,
            "signals": signals,
            "alert": alert,
            "forecast": forecast,
            "diagnosis": diagnosis,
        }
        if "prediction" not in enriched:
            enriched["prediction"] = forecast
        monitoring_store.add_event(session, enriched)
        if forward_alert and alert["severity"] in {"warning", "critical"}:
            orchestrator_url = os.getenv(
                "ORCHESTRATOR_URL", "http://127.0.0.1:5002"
            ).rstrip("/")
            try:
                response = requests.post(
                    f"{orchestrator_url}/alerts",
                    json={
                        **alert,
                        "model_name": session.model_name,
                        "model_version": alert.get("model_version"),
                        "signals": signals,
                        "prediction": forecast,
                        "diagnosis": diagnosis,
                        "timestamp": enriched["timestamp"],
                    },
                    timeout=5,
                )
                response.raise_for_status()
                enriched["pipeline"] = response.json()
            except requests.RequestException as exc:
                app.logger.warning("Could not forward alert to orchestrator: %s", exc)
                enriched["pipeline"] = {
                    "status": "forwarding_failed",
                    "message": "The alert was recorded, but the orchestrator did not confirm receipt.",
                }
        return enriched

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "model-monitoring",
        })

    @app.get("/dashboard")
    def dashboard() -> Response:
        response = app.make_response(render_template("dashboard.html"))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/dashboard")
    def dashboard_data():
        try:
            analytics = dashboard_analytics(
                monitoring_store.snapshots(),
                request.args.get("session_id") or None,
            )
        except KeyError:
            return jsonify({
                "error": "monitoring session not found"
            }), 404

        jobs = hyperparameter_tuning_store.public_jobs()

        analytics["tuning_jobs"] = [
            {
                "id": job["id"],
                "model_name": job["model_name"],
                "metric_name": job["metric_name"],
                "direction": job["direction"],
                "status": job["status"],
                "trials": job["trials"],
                "best_trial": job["best_trial"],
            }
            for job in jobs
        ]

        analytics["summary"]["tuning_job_count"] = len(jobs)
        analytics["tuning_figure"] = tuning_figure(jobs)
        analytics["fraud_model"] = (
            fraud_detector.public() if fraud_detector is not None else None
        )
        return jsonify(analytics)

    @app.post("/api/dashboard/insights")
    def dashboard_insights():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "error": "request body must be a JSON object"
            }), 400

        session_id = body.get("session_id")

        if session_id is not None and not isinstance(session_id, str):
            return jsonify({
                "error": "session_id must be a string"
            }), 400

        try:
            analytics = dashboard_analytics(
                monitoring_store.snapshots(),
                session_id or None,
            )
        except KeyError:
            return jsonify({
                "error": "monitoring session not found"
            }), 404

        summary = {
            **analytics["summary"],
            "active_alerts": analytics["active_alerts"],
            "recent_events": analytics["events"][-50:],
        }
        jobs = hyperparameter_tuning_store.public_jobs()
        explanations = metric_explanations(
            analytics,
            jobs,
            fraud_detector.public() if fraud_detector is not None else None,
        )
        try:
            insight = llm.analyze_dashboard(summary)
        except (AttributeError, requests.RequestException, ValueError) as exc:
            app.logger.warning("Dashboard insight generation unavailable: %s", exc)
            return (
                jsonify(
                    {
                        "error": "LLM dashboard analysis is unavailable",
                        "explanations": explanations,
                    }
                ),
                503,
            )
        return jsonify({"insight": insight, "explanations": explanations})

    @app.post("/monitoring/sessions")
    def create_session():
        body = request.get_json(silent=True) or {}

        model_name = body.get("model_name")

        if not isinstance(model_name, str) or not model_name.strip():
            return jsonify({"error": "model_name is required"}), 400
        baseline = body.get(
            "baseline", fraud_detector.baseline if fraud_detector is not None else {}
        )
        if not isinstance(baseline, dict):
            return jsonify({"error": "baseline must be an object"}), 400
        try:
            normalized: dict[str, Any] = {}
            if "mean" in baseline:
                if not _finite_number(baseline["mean"]):
                    raise ValueError("baseline.mean must be a finite number")
                normalized["mean"] = float(baseline["mean"])
            elif "stddev" in baseline:
                raise ValueError("baseline.mean is required with stddev")
            if "stddev" in baseline:
                if not _finite_number(baseline["stddev"]):
                    raise ValueError("baseline.stddev must be a finite number")
                if float(baseline["stddev"]) < 0:
                    raise ValueError("baseline.stddev must not be negative")
                normalized["stddev"] = float(baseline["stddev"])
            if "metrics" in baseline:
                if not isinstance(baseline["metrics"], dict):
                    raise ValueError(
                        "baseline.metrics must be an object"
                    )

                normalized_metrics = {}

                for name, values in baseline["metrics"].items():
                    if (
                        not isinstance(name, str)
                        or not isinstance(values, dict)
                    ):
                        raise ValueError(
                            "each named baseline must be an object"
                        )

                    mean = values.get("mean")
                    stddev = values.get("stddev", 0)

                    if (
                        not _finite_number(mean)
                        or not _finite_number(stddev)
                        or float(stddev) < 0
                    ):
                        raise ValueError(
                            f"invalid baseline for {name}"
                        )

                    normalized_metrics[name] = {
                        "mean": float(mean),
                        "stddev": float(stddev),
                    }

                normalized["metrics"] = normalized_metrics
            thresholds = baseline.get("_thresholds")
            if thresholds is not None:
                if not isinstance(thresholds, dict):
                    raise ValueError("baseline._thresholds must be an object")
                normalized_thresholds: dict[str, dict[str, float]] = {}
                for name, bounds in thresholds.items():
                    if not isinstance(name, str) or not isinstance(bounds, dict):
                        raise ValueError("each threshold must be an object")
                    normalized_bounds: dict[str, float] = {}
                    for bound in ("min", "max"):
                        if bound in bounds:
                            if not _finite_number(bounds[bound]):
                                raise ValueError("threshold bounds must be finite numbers")
                            normalized_bounds[bound] = float(bounds[bound])
                    if not normalized_bounds:
                        raise ValueError(f"threshold for {name} needs min or max")
                    normalized_thresholds[name] = normalized_bounds
                normalized["_thresholds"] = normalized_thresholds
        except (OverflowError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        session = monitoring_store.create(model_name.strip(), normalized)
        return jsonify(session.public()), 201

    @app.post("/monitoring/sessions/<session_id>/events")
    def ingest_event(session_id: str):
        session = monitoring_store.get(session_id)

        if session is None:
            return jsonify({
                "error": "monitoring session not found"
            }), 404

        event = request.get_json(silent=True)

        if not isinstance(event, dict):
            return jsonify({"error": "event must be a JSON object"}), 400
        values = event.get("features", event.get("prediction"))
        numeric_values = (
            [values]
            if isinstance(values, (int, float)) and not isinstance(values, bool)
            else values
            if isinstance(values, list)
            else []
        )
        if any(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not _finite_number(value)
            for value in numeric_values
        ):
            return jsonify({"error": "event numeric values must be finite numbers"}), 400
        return jsonify(record_event(session, event)), 201

    @app.post("/monitoring/sessions/<session_id>/predict")
    def predict_event(session_id: str) -> tuple[Any, int]:
        session = monitoring_store.get(session_id)
        if session is None:
            return jsonify({"error": "monitoring session not found"}), 404
        if fraud_detector is None:
            return jsonify({"error": "fraud detection model is not loaded"}), 503
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        features = body.get("features")
        if not isinstance(features, dict):
            return jsonify({"error": "features must be an object keyed by column"}), 400
        actual = body.get("actual")
        if actual is not None and (
            isinstance(actual, bool) or actual not in (0, 1)
        ):
            return jsonify({"error": "actual must be the binary label 0 or 1"}), 400
        try:
            prediction = fraud_detector.predict(features)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        event = {
            "features": features,
            "prediction": prediction["fraud_probability"],
            **prediction,
        }
        if actual is not None:
            event["actual"] = actual
        if isinstance(body.get("timestamp"), str):
            event["timestamp"] = body["timestamp"]
        return jsonify(record_event(session, event)), 201

    @app.post("/monitoring/sessions/<session_id>/predict/batch")
    def predict_batch(session_id: str) -> tuple[Any, int]:
        session = monitoring_store.get(session_id)
        if session is None:
            return jsonify({"error": "monitoring session not found"}), 404
        if fraud_detector is None:
            return jsonify({"error": "fraud detection model is not loaded"}), 503
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        rows = body.get("rows")
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "rows must be a non-empty array"}), 400
        if len(rows) > 1000:
            return jsonify({"error": "a batch may contain at most 1000 rows"}), 400
        diagnose = body.get("diagnose", False)
        if not isinstance(diagnose, bool):
            return jsonify({"error": "diagnose must be a boolean"}), 400

        pending_events: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return jsonify({"error": f"rows[{index}] must be an object"}), 400
            features = row.get("features")
            if not isinstance(features, dict):
                return jsonify(
                    {"error": f"rows[{index}].features must be an object"}
                ), 400
            actual = row.get("actual")
            if actual is not None and (
                isinstance(actual, bool) or actual not in (0, 1)
            ):
                return jsonify(
                    {"error": f"rows[{index}].actual must be 0 or 1"}
                ), 400
            try:
                prediction = fraud_detector.predict(features)
            except (TypeError, ValueError) as exc:
                return jsonify({"error": f"rows[{index}]: {exc}"}), 400

            event = {
                "features": features,
                "prediction": prediction["fraud_probability"],
                **prediction,
            }
            if actual is not None:
                event["actual"] = actual
            if isinstance(row.get("timestamp"), str):
                event["timestamp"] = row["timestamp"]
            pending_events.append(event)

        events = [
            record_event(
                session,
                event,
                run_diagnosis=diagnose,
                forward_alert=False,
            )
            for event in pending_events
        ]
        return jsonify({"session_id": session_id, "count": len(events), "events": events}), 201

    @app.get("/monitoring/sessions/<session_id>/stream")
    def stream_events(session_id: str):
        session = monitoring_store.get(session_id)

        if session is None:
            return jsonify({
                "error": "monitoring session not found"
            }), 404

        subscriber = monitoring_store.subscribe(session)

        def generate() -> Iterator[str]:
            try:
                yield "event: ready\ndata: {}\n\n"

                while True:
                    event = subscriber.get()
                    yield f"data: {json.dumps(event)}\n\n"

            finally:
                monitoring_store.unsubscribe(
                    session,
                    subscriber,
                )

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/monitoring/sessions/<session_id>")
    def get_session(session_id: str):
        session = monitoring_store.get(session_id)

        if session is None:
            return jsonify({
                "error": "monitoring session not found"
            }), 404

        return jsonify(session.public())

    @app.get("/api/alerts")
    def list_alerts():
        session_id = request.args.get("session_id")

        try:
            analytics = dashboard_analytics(
                monitoring_store.snapshots(),
                session_id or None,
            )
        except KeyError:
            return jsonify({
                "error": "monitoring session not found"
            }), 404

        alerts = [
            {
                **event["alert"],
                "timestamp": event.get("timestamp"),
                "diagnosis": event.get("diagnosis"),
                "signals": event.get("signals"),
            }
            for event in analytics["events"]
            if isinstance(event.get("alert"), dict)
            and event["alert"].get("severity") != "info"
        ]

        return jsonify({"alerts": alerts})

    @app.post("/tuning/jobs")
    def create_tuning_job():
        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "error": "request body must be a JSON object"
            }), 400

        model_name = body.get("model_name")
        metric_name = body.get("metric_name")
        direction = body.get("direction")

        if not isinstance(model_name, str) or not model_name.strip():
            return jsonify({"error": "model_name is required"}), 400

        if not isinstance(metric_name, str) or not metric_name.strip():
            return jsonify({"error": "metric_name is required"}), 400

        if direction not in {"minimize", "maximize"}:
            return jsonify({
                "error": "direction must be minimize or maximize"
            }), 400

        max_trials = body.get("max_trials", 10)

        if type(max_trials) is not int or not 1 <= max_trials <= 100:
            return jsonify({
                "error": "max_trials must be an integer between 1 and 100"
            }), 400

        try:
            search_space = normalize_search_space(
                body.get("search_space")
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            proposed = llm.generate_candidates(
                search_space,
                max_trials,
            )
        except (requests.RequestException, ValueError) as exc:
            app.logger.warning(
                "Ollama candidate generation unavailable: %s",
                exc,
            )
            proposed = None

        try:
            job = hyperparameter_tuning_store.create(
                model_name=model_name.strip(),
                search_space=search_space,
                metric_name=metric_name.strip(),
                direction=direction,
                max_trials=max_trials,
                proposed=proposed,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        public_job = job.public()
        sources = {candidate["source"] for candidate in public_job["candidates"]}
        if "fallback" in sources:
            app.logger.warning(
                "Tuning job %s used fallback candidates for some or all trials",
                job.id,
            )
        return jsonify(public_job), 201

    @app.get("/tuning/jobs/<job_id>")
    def get_tuning_job(job_id: str):
        job = hyperparameter_tuning_store.get(job_id)

        if job is None:
            return jsonify({
                "error": "tuning job not found"
            }), 404

        return jsonify(job.public())

    @app.post("/tuning/jobs/<job_id>/trials")
    def submit_tuning_trial(job_id: str):
        job = hyperparameter_tuning_store.get(job_id)

        if job is None:
            return jsonify({
                "error": "tuning job not found"
            }), 404

        body = request.get_json(silent=True)

        if not isinstance(body, dict):
            return jsonify({
                "error": "request body must be a JSON object"
            }), 400

        candidate_id = body.get("candidate_id")
        value = body.get("metric_value")

        if not isinstance(candidate_id, str) or not candidate_id.strip():
            return jsonify({
                "error": "candidate_id is required"
            }), 400

        if not _finite_number(value):
            return jsonify({
                "error": "metric_value must be a finite number"
            }), 400

        try:
            hyperparameter_tuning_store.submit_trial(
                job,
                candidate_id.strip(),
                value,
            )
        except KeyError:
            return jsonify({
                "error": "candidate not found"
            }), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        return jsonify(job.public()), 201

    return app


_configured_fraud_model_path = os.getenv(
    "FRAUD_MODEL_PATH", "artifacts/fraud_detector.joblib"
)
_configured_fraud_detector = (
    load_fraud_model(_configured_fraud_model_path)
    if os.path.isfile(_configured_fraud_model_path)
    else None
)
app = create_app(fraud_detector=_configured_fraud_detector)


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
