import unittest

import requests

from app import create_app


class FakeAnalyzer:
    def __init__(self, candidates=None, candidate_error=None):
        self.candidates = candidates
        self.candidate_error = candidate_error

    def analyze(self, event, signals):
        return "Monitoring diagnosis"

    def generate_candidates(self, search_space, count):
        if self.candidate_error:
            raise self.candidate_error
        return self.candidates


class TuningApiTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = FakeAnalyzer(
            candidates=[
                {"n_estimators": 100, "max_depth": 3},
                {"n_estimators": 1000, "max_depth": 3},
            ]
        )
        self.client = create_app(analyzer=self.analyzer).test_client()
        self.job_payload = {
            "model_name": "fraud-detector",
            "metric_name": "validation_auc",
            "direction": "maximize",
            "max_trials": 3,
            "search_space": {
                "n_estimators": {"type": "int", "min": 50, "max": 200},
                "max_depth": [3, 5],
            },
        }

    def test_job_filters_invalid_suggestions_and_fills_from_fallback(self):
        response = self.client.post("/tuning/jobs", json=self.job_payload)
        self.assertEqual(response.status_code, 201)
        job = response.get_json()
        self.assertEqual(len(job["candidates"]), 3)
        self.assertEqual(job["candidates"][0]["source"], "ollama")
        self.assertEqual(job["candidates"][1]["source"], "fallback")
        self.assertTrue(
            all(
                50 <= candidate["config"]["n_estimators"] <= 200
                and candidate["config"]["max_depth"] in [3, 5]
                for candidate in job["candidates"]
            )
        )

    def test_trials_track_metric_and_select_best(self):
        created = self.client.post("/tuning/jobs", json=self.job_payload).get_json()
        for candidate_id, score in (
            ("candidate-1", 0.81),
            ("candidate-2", 0.95),
        ):
            response = self.client.post(
                f"/tuning/jobs/{created['id']}/trials",
                json={"candidate_id": candidate_id, "metric_value": score},
            )
            self.assertEqual(response.status_code, 201)
        job = self.client.get(f"/tuning/jobs/{created['id']}").get_json()
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["best_trial"]["candidate_id"], "candidate-2")
        self.assertEqual(job["best_trial"]["metric_value"], 0.95)

        duplicate = self.client.post(
            f"/tuning/jobs/{created['id']}/trials",
            json={"candidate_id": "candidate-2", "metric_value": 0.99},
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_unavailable_ollama_uses_fallback_candidates(self):
        self.client = create_app(
            analyzer=FakeAnalyzer(candidate_error=requests.ConnectionError("offline"))
        ).test_client()
        response = self.client.post("/tuning/jobs", json=self.job_payload)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(
            all(
                candidate["source"] == "fallback"
                for candidate in response.get_json()["candidates"]
            )
        )

    def test_malformed_ollama_output_uses_fallback_candidates(self):
        self.client = create_app(
            analyzer=FakeAnalyzer(candidate_error=ValueError("malformed JSON"))
        ).test_client()
        response = self.client.post("/tuning/jobs", json=self.job_payload)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(
            all(
                candidate["source"] == "fallback"
                for candidate in response.get_json()["candidates"]
            )
        )

    def test_invalid_space_and_metric_are_rejected(self):
        invalid_space = {
            **self.job_payload,
            "search_space": {"n": {"type": "int", "min": 4, "max": 1}},
        }
        self.assertEqual(
            self.client.post("/tuning/jobs", json=invalid_space).status_code, 400
        )
        invalid_metric = {
            "candidate_id": "candidate-1",
            "metric_value": float("nan"),
        }
        created = self.client.post("/tuning/jobs", json=self.job_payload).get_json()
        self.assertEqual(
            self.client.post(
                f"/tuning/jobs/{created['id']}/trials", json=invalid_metric
            ).status_code,
            400,
        )

    def test_existing_monitoring_ingestion_still_works(self):
        session = self.client.post(
            "/monitoring/sessions",
            json={"model_name": "fraud-detector", "baseline": {"mean": 0.5}},
        ).get_json()
        response = self.client.post(
            f"/monitoring/sessions/{session['id']}/events",
            json={"features": [0.7]},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["diagnosis"], "Monitoring diagnosis")
        self.assertIn("signals", response.get_json())


if __name__ == "__main__":
    unittest.main()
