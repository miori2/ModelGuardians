
"""Ollama client for monitoring analysis and tuning candidates."""

from __future__ import annotations

import json
from typing import Any

import requests


class OllamaAnalyzer:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 10.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _generate(
        self,
        prompt: str,
        json_mode: bool = False,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if json_mode:
            body["format"] = "json"

        response = requests.post(
            f"{self.host}/api/generate",
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        answer = payload.get("response")

        if not isinstance(answer, str):
            raise ValueError("Ollama returned a malformed response")

        return answer.strip()

    def analyze(
        self,
        event: dict[str, Any],
        signals: dict[str, Any],
    ) -> str:
        prompt = (
            "Analyze this model monitoring event. Explain only what "
            "the evidence supports, identify possible anomaly or drift "
            "causes, and recommend one investigation step. "
            "Do not authorize system actions. Keep it concise.\n\n"
            f"Event: {json.dumps(event, sort_keys=True)}\n"
            f"Signals: {json.dumps(signals, sort_keys=True)}"
        )

        return self._generate(prompt)

    def assess_alert(
        self,
        event: dict[str, Any],
        signals: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = (
            "Assess the supplied model monitoring evidence. "
            "Do not invent metrics. A forecast is uncertain. "
            "Do not authorize deployments, rollbacks, or other "
            "system actions. Return JSON only with exactly these fields: "
            "risk (low, medium, high, unknown), "
            "explanation (string), "
            "recommended_action "
            "(investigate, validate_model, pause_promotion, none). "
            "If evidence is insufficient, use risk unknown. "
            "Recommendations are advisory.\n\n"
            f"Event: {json.dumps(event, sort_keys=True)}\n"
            f"Signals: {json.dumps(signals, sort_keys=True)}\n"
            f"Forecast: {json.dumps(prediction, sort_keys=True)}"
        )

        raw = self._generate(prompt, json_mode=True)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ollama did not return valid JSON"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Ollama assessment must be a JSON object"
            )

        if result.get("risk") not in {
            "low", "medium", "high", "unknown"
        }:
            raise ValueError("Ollama returned an invalid risk value")

        if result.get("recommended_action") not in {
            "investigate",
            "validate_model",
            "pause_promotion",
            "none",
        }:
            raise ValueError(
                "Ollama returned an invalid recommended action"
            )

        if not isinstance(result.get("explanation"), str):
            raise ValueError(
                "Ollama explanation must be a string"
            )

        return result

    def analyze_dashboard(
        self,
        summary: dict[str, Any],
    ) -> str:
        prompt = (
            "Summarize these aggregated machine-learning monitoring "
            "metrics. Explain notable trends and anomalies. Treat "
            "linear forecasts as uncertain. State when there is "
            "insufficient data. Do not authorize system actions.\n\n"
            f"Dashboard metrics: {json.dumps(summary, sort_keys=True)}"
        )

        return self._generate(prompt)

    def generate_candidates(
        self,
        search_space: dict[str, Any],
        count: int,
    ) -> list[dict[str, Any]]:
        prompt = (
            "Generate hyperparameter configurations for this search "
            f"space. Return only a JSON array with exactly {count} "
            "objects. Each object must use only supplied parameter "
            "names and values within their declared domains. "
            "Do not include markdown or explanations.\n\n"
            f"Search space: {json.dumps(search_space, sort_keys=True)}"
        )

        raw = self._generate(prompt, json_mode=True)

        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ollama returned invalid candidate JSON"
            ) from exc

        if not isinstance(candidates, list):
            raise ValueError(
                "Ollama candidate response must be a JSON array"
            )

        return candidates
