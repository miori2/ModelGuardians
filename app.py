
"""Flask application for monitoring, forecasting, Ollama insights, and alerts."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _forecast_window(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[float] = []
    for event in events:
        observed = event.get("signals", {}).get("observed_mean")
        if _finite(observed):
            values.append(float(observed))

    if not values:
        return []

    current = max(values[-1], values[-1] * 1.05 if values[-1] >= 0 else values[-1] * 0.95)
    slope = 0.0
    if len(values) > 1:
        slope = values[-1] - values[-2]

    forecast = []
    for offset in range(3):
        forecast_mean = current + slope * (offset + 1)
        forecast.append(
            {
                "step": offset + 1,
                "observed_mean": current if offset == 0 else current + slope * offset,
                "forecast_mean": forecast_mean,
            }
        )

    return forecast


def _monitoring_figure(events: list[dict[str, Any]]) -> dict[str, Any]:
    x_values = []
    observed = []
    for idx, event in enumerate(events):
        observed_mean = event.get("signals", {}).get("observed_mean")
        if _finite(observed_mean):
            x_values.append(str(idx + 1))
            observed.append(float(observed_mean))

    if not observed:
        return {"data": [], "layout": {"title": "Observed signal and short-term forecast"}}

    trend = observed[:]
    if len(trend) > 1:
        slope = trend[-1] - trend[-2]
        trend += [trend[-1] + slope * (step + 1) for step in range(3)]
        x_forecast = [*x_values, *[str(len(x_values) + step) for step in range(1, 4)]]
    else:
        trend = observed + [observed[-1]] * 3
        x_forecast = [*x_values, *[str(len(x_values) + step) for step in range(1, 4)]]

    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Observed mean",
                "x": x_values,
                "y": observed,
            },
            {
                "type": "scatter",
                "mode": "lines",
                "name": "Linear forecast",
                "x": x_forecast,
                "y": trend,
            },
        ],
        "layout": {
            "title": "Observed signal and short-term forecast",
            "height": 320,
        },
    }


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

    predictions = [
        event.get("prediction", {})
        for event in events
    ]

    forecast = _forecast_window(events[:30])

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
        "monitoring_figure": _monitoring_figure(events[-30:]),
        "tuning_figure": {"data": [{"type": "bar", "x": [], "y": []}], "layout": {"title": "Tuning jobs"}},
        "summary": {
            "session_count": len(active_sessions),
            "event_count": len(events),
            "anomaly_count": anomalies,
            "drift_count": drifts,
            "active_alert_count": len(active_alerts),
            "predictive_warning_count": sum(
                1
                for prediction in predictions
                if isinstance(prediction, dict)
                and any(
                    isinstance(metric, dict)
                    and metric.get("predicted_breach") is True
                    for metric in prediction.get("metrics", {}).values()
                )
            ),
            "model_count": len(
                {session["model_name"] for session in active_sessions}
            ),
            "tuning_job_count": 0,
        },
        "active_alerts": active_alerts[-100:],
    }


def create_app(
    analyzer: OllamaAnalyzer | None = None,
    store: MonitoringStore | None = None,
    tuning_store: TuningStore | None = None,
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

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "model-monitoring",
        })

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

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
        analytics["tuning_figure"] = {
            "data": [
                {
                    "type": "bar",
                    "x": [job["model_name"] for job in jobs],
                    "y": [len(job["trials"]) for job in jobs],
                    "name": "Trial count",
                }
            ],
            "layout": {"title": "Tuning jobs"},
        }

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

        try:
            if not hasattr(llm, "analyze_dashboard"):
                raise AttributeError("Analyzer does not support dashboard summaries")

            insight = llm.analyze_dashboard(summary)
            return jsonify({"insight": insight})

        except (AttributeError, requests.RequestException, ValueError) as exc:
            app.logger.warning(
                "Dashboard insight generation unavailable: %s",
                exc,
            )

            return jsonify({
                "error": "LLM dashboard analysis is unavailable"
            }), 503

    @app.post("/monitoring/sessions")
    def create_session():
        body = request.get_json(silent=True) or {}

        model_name = body.get("model_name")

        if not isinstance(model_name, str) or not model_name.strip():
            return jsonify({
                "error": "model_name is required"
            }), 400

        baseline = body.get("baseline", {})

        if not isinstance(baseline, dict):
            return jsonify({
                "error": "baseline must be an object"
            }), 400

        # Accept named baselines or the legacy mean/stddev format.
        try:
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
                        not _finite(mean)
                        or not _finite(stddev)
                        or float(stddev) < 0
                    ):
                        raise ValueError(
                            f"invalid baseline for {name}"
                        )

                    normalized_metrics[name] = {
                        "mean": float(mean),
                        "stddev": float(stddev),
                    }

                normalized = {
                    "metrics": normalized_metrics
                }

            elif baseline:
                mean = baseline.get("mean")
                stddev = baseline.get("stddev", 0)

                if (
                    not _finite(mean)
                    or not _finite(stddev)
                    or float(stddev) < 0
                ):
                    raise ValueError(
                        "baseline mean/stddev must be finite numbers "
                        "and stddev non-negative"
                    )

                normalized = {
                    "mean": float(mean),
                    "stddev": float(stddev),
                }

            else:
                normalized = {}

        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

        thresholds = body.get("thresholds", {})

        if not isinstance(thresholds, dict):
            return jsonify({
                "error": "thresholds must be an object"
            }), 400

        session = monitoring_store.create(
            model_name.strip(),
            normalized,
        )

        # Prototype: keep thresholds with the session baseline.
        session.baseline["_thresholds"] = thresholds

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
            return jsonify({
                "error": "event must be a JSON object"
            }), 400

        metrics = event.get("metrics")

        if metrics is not None:
            if not isinstance(metrics, dict):
                return jsonify({
                    "error": "metrics must be an object"
                }), 400

            for name, value in metrics.items():
                if not isinstance(name, str) or not _finite(value):
                    return jsonify({
                        "error": (
                            f"metric {name!r} must be a finite number"
                        )
                    }), 400

        # 1. Detect anomalies using Python statistical rules.
        signals = evaluate_event(event, session.baseline)

        # 2. Predict possible future threshold violations.
        thresholds = session.baseline.get("_thresholds", {})

        predictions = forecast_metrics(
            session.metric_history,
            thresholds=thresholds,
            horizon=5,
            min_points=4,
        )

        prediction = {"metrics": predictions}

        version = event.get("model_version")

        # 3. Create a deterministic alert.
        alert = evaluate_alert(
            signals,
            prediction,
            model_name=session.model_name,
            model_version=(
                version if isinstance(version, str) else None
            ),
        )

        # 4. Ask Ollama to explain the evidence.
        # Its response is advisory, not a deployment command.
        try:
            if hasattr(llm, "assess_alert"):
                assessment = llm.assess_alert(
                    event,
                    signals,
                    prediction,
                )
            elif hasattr(llm, "analyze"):
                assessment = llm.analyze(event, signals)
            else:
                assessment = (
                    "Monitoring assessment unavailable; use the "
                    "statistical signals and forecast."
                )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            AttributeError,
        ) as exc:
            app.logger.warning(
                "Ollama assessment unavailable: %s",
                exc,
            )

            assessment = (
                "Ollama assessment unavailable; use the "
                "statistical signals and forecast."
            )

        # 5. Combine the results for the dashboard and API clients.
        enriched = {
            **event,
            "timestamp": event.get("timestamp", utc_now()),
            "model_name": session.model_name,
            "signals": signals,
            "prediction": prediction,
            "alert": alert,
            "diagnosis": assessment,
        }

        
        # Save the event for the dashboard and event stream.
        monitoring_store.add_event(session, enriched)

        # Forward actionable alerts to the deployment orchestrator.
        orchestrator_url = os.getenv(
            "ORCHESTRATOR_URL",
            "http://127.0.0.1:5002",
        ).rstrip("/")

        if alert["severity"] in {"warning", "critical"}:
            try:
                pipeline_response = requests.post(
                    f"{orchestrator_url}/alerts",
                    json={
                        **alert,
                        "model_name": session.model_name,
                        "model_version": alert.get("model_version"),
                        "signals": signals,
                        "prediction": prediction,
                        "diagnosis": assessment,
                        "timestamp": enriched["timestamp"],
                    },
                    timeout=5,
                )

                pipeline_response.raise_for_status()

                enriched["pipeline"] = pipeline_response.json()

            except requests.RequestException as exc:
                # Keep the monitoring event even if the orchestrator
                # is unavailable. Report that forwarding failed.
                app.logger.warning(
                    "Could not forward alert to orchestrator: %s",
                    exc,
                )

                enriched["pipeline"] = {
                    "status": "forwarding_failed",
                    "message": (
                        "The alert was recorded by monitoring, "
                        "but the orchestrator did not confirm receipt."
                    ),
                }

        return jsonify(enriched), 201

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

        job = hyperparameter_tuning_store.create(
            model_name.strip(),
            search_space,
            metric_name.strip(),
            direction,
            max_trials,
            proposed,
        )

        return jsonify(job.public()), 201

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

        if not _finite(value):
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


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
