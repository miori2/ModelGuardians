import json
import unittest
from unittest.mock import Mock, patch

import requests

from ollama_client import OllamaAnalyzer


class OllamaAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = OllamaAnalyzer(host="http://ollama.test", model="test-model")

    @staticmethod
    def response(*chunks):
        response = Mock()
        response.iter_lines.return_value = [
            json.dumps(chunk).encode() for chunk in chunks
        ]
        return response

    @patch("ollama_client.requests.post")
    def test_generate_candidates_parses_streamed_json(self, post):
        post.return_value = self.response(
            {"response": '[{"depth": 3}', "done": False},
            {"response": ']', "done": True},
        )
        result = self.analyzer.generate_candidates({"depth": [3]}, 1)
        self.assertEqual(result, [{"depth": 3}])
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["timeout"], 10.0)

    @patch("ollama_client.requests.post")
    def test_generation_rejects_malformed_json_and_chunks(self, post):
        post.return_value = self.response({"response": "not-json", "done": True})
        with self.assertRaises(json.JSONDecodeError):
            self.analyzer.generate_candidates({"depth": [3]}, 1)

        post.return_value = self.response(["wrong chunk shape"])
        with self.assertRaises(ValueError):
            self.analyzer.generate_candidates({"depth": [3]}, 1)

    @patch("ollama_client.requests.post")
    def test_transport_and_http_errors_are_propagated(self, post):
        post.side_effect = requests.ConnectionError("offline")
        with self.assertRaises(requests.ConnectionError):
            self.analyzer.generate_candidates({"depth": [3]}, 1)

        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("server error")
        post.side_effect = None
        post.return_value = response
        with self.assertRaises(requests.HTTPError):
            self.analyzer.analyze({"prediction": 1}, {"anomaly": False})

    @patch("ollama_client.requests.post")
    def test_analysis_returns_joined_stream(self, post):
        post.return_value = self.response(
            {"response": "Review ", "done": False},
            {"response": "the signal.", "done": True},
        )
        self.assertEqual(
            self.analyzer.analyze({"prediction": 1}, {"anomaly": False}),
            "Review the signal.",
        )


if __name__ == "__main__":
    unittest.main()
