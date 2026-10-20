import unittest

import requests

from app import create_app


class DashboardAnalyzer:
    def __init__(self, insight_error=None):
        self.insight_error = insight_error
        self.summary = None

    def analyze(self, event, signals):
        return "Monitoring diagnosis"

    def generate_candidates(self, search_space, count):
        return None

    def analyze_dashboard(self, summary):
        if self.insight_error:
            raise self.insight_error
        self.summary = summary
        return "The observed signal is trending upward."


class DashboardApiTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = DashboardAnalyzer()
        self.client = create_app(analyzer=self.analyzer).test_client()
        self.session = self.client.post(
            "/monitoring/sessions",
            json={
                "model_name": "fraud-detector",
                "baseline": {"mean": 0.5, "stddev": 0.1},
            },
        ).get_json()
        for value in (0.5, 0.6, 0.8):
            response = self.client.post(
                f"/monitoring/sessions/{self.session['id']}/events",
                json={"features": [value]},
            )
            self.assertEqual(response.status_code, 201)

    def test_dashboard_page_loads(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Observed signal and short-term forecast", response.data)
        self.assertIn(b"Generate insights", response.data)

    def test_dashboard_api_reports_signals_and_forecast(self):
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["summary"]["session_count"], 1)
        self.assertEqual(data["summary"]["event_count"], 3)
        self.assertEqual(data["summary"]["anomaly_count"], 1)
        self.assertEqual(data["summary"]["drift_count"], 1)
        self.assertEqual(len(data["forecast"]), 3)
        self.assertGreater(data["forecast"][0]["observed_mean"], 0.8)
        self.assertEqual(data["sessions"][0]["id"], self.session["id"])
        self.assertEqual(
            data["monitoring_figure"]["data"][0]["name"], "Observed mean"
        )
        self.assertEqual(
            data["monitoring_figure"]["data"][1]["name"], "Linear forecast"
        )
        self.assertEqual(data["tuning_figure"]["data"][0]["type"], "bar")

    def test_dashboard_can_be_scoped_and_rejects_unknown_session(self):
        response = self.client.get(
            f"/api/dashboard?session_id={self.session['id']}"
        )
        self.assertEqual(response.get_json()["summary"]["event_count"], 3)
        self.assertEqual(
            self.client.get("/api/dashboard?session_id=missing").status_code, 404
        )

    def test_llm_insights_receive_aggregated_dashboard_metrics(self):
        response = self.client.post(
            "/api/dashboard/insights", json={"session_id": self.session["id"]}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("upward", response.get_json()["insight"])
        self.assertEqual(self.analyzer.summary["event_count"], 3)
        self.assertEqual(len(self.analyzer.summary["recent_events"]), 3)

    def test_insights_report_unavailable_llm(self):
        client = create_app(
            analyzer=DashboardAnalyzer(
                insight_error=requests.ConnectionError("offline")
            )
        ).test_client()
        response = client.post("/api/dashboard/insights", json={})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json()["error"], "LLM dashboard analysis is unavailable"
        )
        self.assertTrue(response.get_json()["explanations"])

    def test_insights_include_metric_explanations(self):
        data = self.client.post(
            "/api/dashboard/insights", json={"session_id": self.session["id"]}
        ).get_json()
        metrics = {item["metric"]: item for item in data["explanations"]}
        self.assertEqual(metrics["Observed events"]["value"], "3")
        self.assertIn("Anomalies", metrics)
        self.assertIn("Drift signals", metrics)
        self.assertTrue(all(item["explanation"] for item in data["explanations"]))


if __name__ == "__main__":
    unittest.main()
