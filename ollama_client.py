"""Ollama streaming diagnosis client."""

from __future__ import annotations

import json
from typing import Any

import requests


class OllamaAnalyzer:
    """Streams a diagnosis from Ollama's /api/generate endpoint."""

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
            parts.append(payload.get("response", ""))
            if payload.get("done"):
                break
        return "".join(parts).strip()
