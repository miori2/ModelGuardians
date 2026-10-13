"""Flask application for real-time model monitoring."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterator

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

from monitoring import MonitoringStore, evaluate_event
from ollama_client import OllamaAnalyzer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    analyzer: OllamaAnalyzer | None = None,
    store: MonitoringStore | None = None,
) -> Flask:
    app = Flask(__name__)
    monitoring_store = store or MonitoringStore()
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

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
