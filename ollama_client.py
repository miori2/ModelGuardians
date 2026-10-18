"""Ollama streaming diagnosis client."""

from __future__ import annotations

import json
from typing import Any

import requests


class OllamaAnalyzer:
    """Streams diagnoses and candidate configs from Ollama's /api/generate endpoint."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 10.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def analyze(self, event: dict[str, Any], signals: dict[str, Any]) -> str:
        prompt = (
            "Analyze this model monitoring event. Identify likely anomalies or "
            "data drift, explain the evidence, and recommend one action. "
            "Return concise plain text.\n\n"
            f"Event: {json.dumps(event, sort_keys=True)}\n"
            f"Signals: {json.dumps(signals, sort_keys=True)}"
        )
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("response", ""), str
            ):
                raise ValueError("Ollama returned a malformed generation chunk")
            parts.append(payload.get("response", ""))
            if payload.get("done"):
                break
        return "".join(parts).strip()

    def analyze_dashboard(self, summary: dict[str, Any]) -> str:
        prompt = (
                "Analyze these aggregated machine-learning model monitoring "
                "dashboard metrics. Explain notable trends and anomalies, describe "
                "what the short-term linear forecast suggests without treating it "
                "as certainty, and recommend actionable next steps. If there is "
                "insufficient data, say so explicitly. Return concise plain text.\n\n"
                f"Dashboard metrics: {json.dumps(summary, sort_keys=True)}"
        )
        response = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
                timeout=self.timeout,
                stream=True,
        )
        response.raise_for_status()
        parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict) or not isinstance(
                    payload.get("response", ""), str
                ):
                    raise ValueError("Ollama returned a malformed generation chunk")
                parts.append(payload.get("response", ""))
                if payload.get("done"):
                    break
        return "".join(parts).strip()

    def generate_candidates(
        self, search_space: dict[str, Any], count: int
    ) -> list[dict[str, Any]]:
        prompt = (
            "Generate hyperparameter configurations for this search space. "
            f"Return only a JSON array with exactly {count} objects. Each object "
            "must use only the supplied parameter names and values within their "
            "declared domains. Do not include markdown or explanations.\n\n"
            f"Search space: {json.dumps(search_space, sort_keys=True)}"
        )
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True},
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        parts: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("response", ""), str
            ):
                raise ValueError("Ollama returned a malformed generation chunk")
            parts.append(payload.get("response", ""))
            if payload.get("done"):
                break
        candidates = json.loads("".join(parts))
        if not isinstance(candidates, list):
            raise ValueError("Ollama candidate response must be a JSON array")
        return candidates