import unittest

from app import create_app


class FakeAnalyzer:
    def analyze(self, event, signals):
        return "Fraud probability analyzed"

    def generate_candidates(self, search_space, count):
        return None

    def analyze_dashboard(self, summary):
        return "Dashboard analyzed"


class FakeFraudModel:
    baseline = {"mean": 0.1, "stddev": 0.05}

    def predict(self, features):
        if "amount" not in features:
            raise ValueError("Missing required feature(s): amount")
        return {
            "fraud_probability": 0.93 if features["amount"] > 100 else 0.02,
            "predicted_class": 1 if features["amount"] > 100 else 0,
        }

    def public(self):
        return {
            "model_name": "credit-card-fraud-detector",
            "feature_count": 1,
            "metrics": {"average_precision": 0.99},
            "dataset_summary": {"row_count": 100},
        }


class FraudMonitoringApiTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(
            analyzer=FakeAnalyzer(),
            fraud_detector=FakeFraudModel(),
        ).test_client()
        response = self.client.post(
            "/monitoring/sessions",
            json={"model_name": "credit-card-fraud-detector"},
        )
        self.assertEqual(response.status_code, 201)
        self.session_id = response.get_json()["id"]

    def test_real_prediction_is_the_observed_dashboard_signal(self):
        response = self.client.post(
            f"/monitoring/sessions/{self.session_id}/predict",
            json={"features": {"amount": 500}, "actual": 1},
        )

        self.assertEqual(response.status_code, 201)
        event = response.get_json()
        self.assertEqual(event["predicted_class"], 1)
        self.assertEqual(event["actual"], 1)
        self.assertEqual(event["signals"]["observed_mean"], 0.93)
        dashboard = self.client.get("/api/dashboard").get_json()
        self.assertEqual(dashboard["events"][0]["signals"]["observed_mean"], 0.93)
        self.assertEqual(dashboard["fraud_model"]["feature_count"], 1)

    def test_batch_prediction_records_events_without_per_row_llm_calls(self):
        response = self.client.post(
            f"/monitoring/sessions/{self.session_id}/predict/batch",
            json={
                "rows": [
                    {"features": {"amount": 500}, "actual": 1},
                    {"features": {"amount": 25}, "actual": 0},
                ]
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["count"], 2)
        self.assertTrue(
            all(
                "skipped" in event["diagnosis"]
                for event in response.get_json()["events"]
            )
        )

    def test_invalid_batch_is_rejected_without_partial_ingestion(self):
        response = self.client.post(
            f"/monitoring/sessions/{self.session_id}/predict/batch",
            json={
                "rows": [
                    {"features": {"amount": 500}},
                    {"features": {"missing": 5}},
                ]
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            self.client.get("/api/dashboard").get_json()["summary"]["event_count"],
            0,
        )

    def test_prediction_requires_a_loaded_model(self):
        client = create_app(analyzer=FakeAnalyzer()).test_client()
        session = client.post(
            "/monitoring/sessions",
            json={"model_name": "fraud-detector"},
        ).get_json()

        response = client.post(
            f"/monitoring/sessions/{session['id']}/predict",
            json={"features": {"amount": 1}},
        )

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
