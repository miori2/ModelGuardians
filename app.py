"""Flask application for real-time model monitoring."""

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

from monitoring import MonitoringStore, evaluate_event
from ollama_client import OllamaAnalyzer
from tuning import TuningStore, normalize_search_space


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def dashboard_analytics(
    sessions: list[dict[str, Any]], selected_session_id: str | None
) -> dict[str, Any]:
    if selected_session_id:
        selected = next(
            (session for session in sessions if session["id"] == selected_session_id),
            None,
        )
        if selected is None:
            raise KeyError("monitoring session not found")
        events = [
            {**event, "session_id": selected["id"], "model_name": selected["model_name"]}
            for event in selected["events"]
        ]
    else:
        events = [
            {**event, "session_id": session["id"], "model_name": session["model_name"]}
            for session in sessions
            for event in session["events"]
        ]

    means = [
        (index, event["signals"]["observed_mean"])
        for index, event in enumerate(events)
        if isinstance(event.get("signals"), dict)
        and isinstance(event["signals"].get("observed_mean"), (int, float))
        and not isinstance(event["signals"]["observed_mean"], bool)
        and math.isfinite(event["signals"]["observed_mean"])
    ]
    forecast: list[dict[str, float | int]] = []
    if len(means) >= 2:
        x_mean = sum(index for index, _ in means) / len(means)
        y_mean = sum(value for _, value in means) / len(means)
        denominator = sum((index - x_mean) ** 2 for index, _ in means)
        slope = (
            sum((index - x_mean) * (value - y_mean) for index, value in means)
            / denominator
            if denominator
            else 0.0
        )
        horizon = min(5, max(len(means), 1))
        forecast = [
            {
                "step": len(events) + offset,
                "observed_mean": y_mean + slope * (len(events) + offset - x_mean),
            }
            for offset in range(horizon)
        ]

    active_sessions = (
        [session for session in sessions if session["id"] == selected_session_id]
        if selected_session_id
        else sessions
    )
    return {
        "sessions": [
            {key: session[key] for key in ("id", "model_name", "event_count")}
            for session in sessions
        ],
        "selected_session_id": selected_session_id,
        "events": events[-500:],
        "forecast": forecast,
        "summary": {
            "session_count": len(active_sessions),
            "event_count": len(events),
            "anomaly_count": sum(
                bool(event.get("signals", {}).get("anomaly")) for event in events
            ),
            "drift_count": sum(
                bool(event.get("signals", {}).get("drift_indicator"))
                for event in events
            ),
            "model_count": len({session["model_name"] for session in active_sessions}),
            "tuning_job_count": 0,
        },
    }


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
            mode="lines+markers",
            name="Observed mean",
            line={"color": "#67a5ff", "width": 2},
            marker={
                "color": [
                    "#ff786f" if event["signals"].get("anomaly") else "#67a5ff"
                    for event in observed
                ]
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
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a27",
        plot_bgcolor="#111a27",
        font={"color": "#c8d5e9"},
        margin={"l": 48, "r": 18, "t": 12, "b": 48},
        xaxis={"title": "Event sequence", "gridcolor": "#263246"},
        yaxis={"title": "Observed mean", "gridcolor": "#263246"},
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    analyzer: OllamaAnalyzer | None = None,
    store: MonitoringStore | None = None,
    tuning_store: TuningStore | None = None,
) -> Flask:
    app = Flask(__name__)
    monitoring_store = store or MonitoringStore()
    hyperparameter_tuning_store = tuning_store or TuningStore()
    llm_analyzer = analyzer or OllamaAnalyzer(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    )

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok", "service": "model-monitoring"})

    @app.get("/dashboard")
    def dashboard() -> str:
        return render_template("dashboard.html")

    @app.get("/api/dashboard")
    def dashboard_data() -> tuple[Any, int]:
        sessions = monitoring_store.snapshots()
        session_id = request.args.get("session_id") or None
        try:
            analytics = dashboard_analytics(sessions, session_id)
        except KeyError:
            return jsonify({"error": "monitoring session not found"}), 404
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
        analytics["monitoring_figure"] = monitoring_figure(
            analytics["events"], analytics["forecast"]
        )
        analytics["tuning_figure"] = tuning_figure(jobs)
        analytics["summary"]["tuning_job_count"] = len(jobs)
        return jsonify(analytics)

    @app.post("/api/dashboard/insights")
    def dashboard_insights() -> tuple[Any, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        session_id = body.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            return jsonify({"error": "session_id must be a string"}), 400
        sessions = monitoring_store.snapshots()
        try:
            analytics = dashboard_analytics(sessions, session_id or None)
        except KeyError:
            return jsonify({"error": "monitoring session not found"}), 404
        jobs = hyperparameter_tuning_store.public_jobs()
        summary = {
            **analytics["summary"],
            "tuning_job_count": len(jobs),
            "forecast": analytics["forecast"],
            "recent_events": [
                {
                    "timestamp": event.get("timestamp"),
                    "model_name": event["model_name"],
                    "signals": event.get("signals", {}),
                }
                for event in analytics["events"][-100:]
            ],
            "tuning_jobs": [
                {
                    "model_name": job["model_name"],
                    "metric_name": job["metric_name"],
                    "direction": job["direction"],
                    "status": job["status"],
                    "best_trial": job["best_trial"],
                }
                for job in jobs
            ],
        }
        try:
            insight = llm_analyzer.analyze_dashboard(summary)
        except (requests.RequestException, ValueError) as exc:
            app.logger.warning("Dashboard insight generation unavailable: %s", exc)
            return jsonify({"error": "LLM dashboard analysis is unavailable"}), 503
        return jsonify({"insight": insight})

    @app.post("/monitoring/sessions")
    def create_session() -> tuple[Any, int]:
        body = request.get_json(silent=True) or {}
        model_name = body.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            return jsonify({"error": "model_name is required"}), 400
        baseline = body.get("baseline", {})
        if not isinstance(baseline, dict):
            return jsonify({"error": "baseline must be an object"}), 400
        if baseline and "mean" not in baseline:
            return jsonify({"error": "baseline.mean is required"}), 400
        if any(
            isinstance(baseline.get(key), bool)
            for key in ("mean", "stddev")
            if key in baseline
        ):
            return jsonify({"error": "baseline mean and stddev must be numbers"}), 400
        try:
            normalized = (
                {
                    "mean": float(baseline["mean"]),
                    "stddev": float(baseline.get("stddev", 0)),
                }
                if baseline
                else {}
            )
        except (OverflowError, TypeError, ValueError):
            return jsonify({"error": "baseline mean and stddev must be numbers"}), 400
        if not all(math.isfinite(value) for value in normalized.values()):
            return jsonify({"error": "baseline mean and stddev must be finite numbers"}), 400
        if normalized.get("stddev", 0) < 0:
            return jsonify({"error": "baseline.stddev must not be negative"}), 400
        session = monitoring_store.create(model_name.strip(), normalized)
        return jsonify(session.public()), 201

    @app.post("/monitoring/sessions/<session_id>/events")
    def ingest_event(session_id: str) -> tuple[Any, int]:
        session = monitoring_store.get(session_id)
        if session is None:
            return jsonify({"error": "monitoring session not found"}), 404
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
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and not _finite_number(value)
            for value in numeric_values
        ):
            return jsonify({"error": "event numeric values must be finite numbers"}), 400
        signals = evaluate_event(event, session.baseline)
        try:
            diagnosis = llm_analyzer.analyze(event, signals)
        except requests.RequestException as exc:
            app.logger.warning("Ollama analysis unavailable: %s", exc)
            diagnosis = "Ollama analysis unavailable; review the statistical signals."
        enriched = {
            **event,
            "timestamp": event.get("timestamp", utc_now()),
            "signals": signals,
            "diagnosis": diagnosis,
        }
        monitoring_store.add_event(session, enriched)
        return jsonify(enriched), 201

    @app.get("/monitoring/sessions/<session_id>/stream")
    def stream_events(session_id: str) -> Response:
        session = monitoring_store.get(session_id)
        if session is None:
            return jsonify({"error": "monitoring session not found"}), 404
        subscriber = monitoring_store.subscribe(session)

        def generate() -> Iterator[str]:
            try:
                yield "event: ready\ndata: {}\n\n"
                while True:
                    event = subscriber.get()
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                monitoring_store.unsubscribe(session, subscriber)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/monitoring/sessions/<session_id>")
    def get_session(session_id: str) -> tuple[Any, int]:
        session = monitoring_store.get(session_id)
        if session is None:
            return jsonify({"error": "monitoring session not found"}), 404
        return jsonify(session.public())

    @app.post("/tuning/jobs")
    def create_tuning_job() -> tuple[Any, int]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        model_name = body.get("model_name")
        metric_name = body.get("metric_name")
        direction = body.get("direction")
        if not isinstance(model_name, str) or not model_name.strip():
            return jsonify({"error": "model_name is required"}), 400
        if not isinstance(metric_name, str) or not metric_name.strip():
            return jsonify({"error": "metric_name is required"}), 400
        if not isinstance(direction, str) or direction not in {
            "minimize",
            "maximize",
        }:
            return jsonify({"error": "direction must be 'minimize' or 'maximize'"}), 400
        max_trials = body.get("max_trials", 10)
        if type(max_trials) is not int or not 1 <= max_trials <= 100:
            return jsonify({"error": "max_trials must be an integer between 1 and 100"}), 400
        try:
            search_space = normalize_search_space(body.get("search_space"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            proposed = llm_analyzer.generate_candidates(search_space, max_trials)
        except (requests.RequestException, ValueError) as exc:
            app.logger.warning(
                "Ollama candidate generation unavailable or malformed: %s", exc
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
    def get_tuning_job(job_id: str) -> tuple[Any, int]:
        job = hyperparameter_tuning_store.get(job_id)
        if job is None:
            return jsonify({"error": "tuning job not found"}), 404
        return jsonify(job.public())

    @app.post("/tuning/jobs/<job_id>/trials")
    def submit_tuning_trial(job_id: str) -> tuple[Any, int]:
        job = hyperparameter_tuning_store.get(job_id)
        if job is None:
            return jsonify({"error": "tuning job not found"}), 404
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        candidate_id = body.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            return jsonify({"error": "candidate_id is required"}), 400
        metric_value = body.get("metric_value")
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or (
                isinstance(metric_value, float)
                and not math.isfinite(metric_value)
            )
        ):
            return jsonify({"error": "metric_value must be a finite number"}), 400
        try:
            hyperparameter_tuning_store.submit_trial(
                job, candidate_id.strip(), metric_value
            )
        except KeyError:
            return jsonify({"error": "candidate not found"}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify(job.public()), 201

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
