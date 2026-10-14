"""Flask application for real-time model monitoring."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

from monitoring import MonitoringStore, evaluate_event
from ollama_client import OllamaAnalyzer
from tuning import TuningStore, normalize_search_space


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
        try:
            normalized = (
                {
                    "mean": float(baseline["mean"]),
                    "stddev": float(baseline.get("stddev", 0)),
                }
                if baseline
                else {}
            )
        except (TypeError, ValueError):
            return jsonify({"error": "baseline mean and stddev must be numbers"}), 400
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

        job = hyperparameter_tuning_store.create(
            model_name=model_name.strip(),
            search_space=search_space,
            metric_name=metric_name.strip(),
            direction=direction,
            max_trials=max_trials,
            proposed=proposed,
        )
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
