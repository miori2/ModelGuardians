
import json
import os
import requests


# Stores Ollama address, model, and timeout for analysis requests
class OllamaAnalyzer:
    def __init__(
        self,
        host="http://localhost:11434",
        model="llama3.2",
        timeout=120,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

# Wrties the instructions and the event/signals to Ollama, and returns the AI's diagnosis
    def analyze(self, event, signals):
        # 1. Prepare the information for the AI
        prompt = (
            "You are an assistant for ModelGuardians, a machine "
            "learning monitoring tool.\n"
            "Analyze the event and its statistical signals.\n"
            "Explain whether there may be an anomaly or data drift.\n"
            "Explain the evidence and recommend one action.\n"
            "Do not claim that an issue is confirmed without evidence.\n"
            "Keep your answer concise.\n\n"
            f"Event: {json.dumps(event)}\n"
            f"Signals: {json.dumps(signals)}"
        )

        # 2. Send the request to Ollama
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )

        # 3. Check for an HTTP error
        response.raise_for_status()

        # 4. Read the AI's answer
        result = response.json()
        diagnosis = result.get("response", "").strip()

        if not diagnosis:
            raise ValueError("Ollama returned an empty response")

        # 5. Return the answer to the caller
        return diagnosis

# Let's us test the client independently of the rest of the system
if __name__ == "__main__":
    analyzer = OllamaAnalyzer(
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    )

    test_event = {
        "prediction": 0.95,
        "features": [0.2, 0.4, 0.8],
    }

    test_signals = {
        "anomaly": True,
        "drift_indicator": True,
        "z_score": 4.5,
    }

    try:
        answer = analyzer.analyze(test_event, test_signals)
        print("Ollama diagnosis:")
        print(answer)
    except (requests.RequestException, ValueError) as exc:
        print(f"Analysis failed: {exc}")
