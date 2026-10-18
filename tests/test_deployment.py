import unittest
from unittest.mock import Mock, patch

import requests

import deployment_orchestrator
import deployment_worker


class DeploymentWorkerTests(unittest.TestCase):
    def setUp(self):
        deployment_worker.deployments.clear()
        self.client = deployment_worker.app.test_client()

    def test_deploy_records_simulated_status_and_lookup(self):
        response = self.client.post(
            "/deploy",
            json={
                "model_name": "fraud-detector",
                "version": "v1",
                "artifact_path": "/tmp/model.bin",
            },
        )
        self.assertEqual(response.status_code, 201)
        deployment = response.get_json()
        self.assertEqual(deployment["status"], "succeeded")
        self.assertIn("simulated", deployment["message"].lower())
        lookup = self.client.get(f"/deployments/{deployment['deployment_id']}")
        self.assertEqual(lookup.status_code, 200)
        self.assertEqual(lookup.get_json(), deployment)

    def test_invalid_deploy_and_unknown_status(self):
        self.assertEqual(self.client.post("/deploy", json={}).status_code, 400)
        self.assertEqual(self.client.get("/deployments/missing").status_code, 404)


class DeploymentOrchestratorTests(unittest.TestCase):
    def setUp(self):
        deployment_orchestrator.models.clear()
        deployment_orchestrator.active_versions.clear()
        deployment_orchestrator.deployment_history.clear()
        self.client = deployment_orchestrator.app.test_client()
        self.register("v1")
        self.register("v2")

    def register(self, version):
        return self.client.post(
            "/models",
            json={
                "model_name": "fraud-detector",
                "version": version,
                "artifact_path": f"/tmp/{version}.bin",
            },
        )

    @staticmethod
    def worker_result(status="succeeded"):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"status": status, "deployment_id": "job-1"}
        return response

    def test_register_duplicate_activate_and_list(self):
        self.assertEqual(self.register("v1").status_code, 409)
        self.assertEqual(
            self.client.get("/models").get_json()["fraud-detector"].keys(),
            {"v1", "v2"},
        )
        with patch(
            "deployment_orchestrator.requests.post",
            return_value=self.worker_result(),
        ) as post:
            response = self.client.post(
                "/deployments",
                json={"model_name": "fraud-detector", "version": "v1"},
            )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["active_version"], "v1")
        self.assertEqual(post.call_args.kwargs["timeout"], 10)
        self.assertEqual(
            self.client.get("/deployments")
            .get_json()["active_versions"]["fraud-detector"],
            "v1",
        )

    def test_unregistered_model_and_worker_failures_do_not_activate(self):
        self.assertEqual(
            self.client.post(
                "/deployments", json={"model_name": "missing", "version": "v1"}
            ).status_code,
            404,
        )
        with patch(
            "deployment_orchestrator.requests.post",
            side_effect=requests.ConnectionError("offline"),
        ):
            response = self.client.post(
                "/deployments",
                json={"model_name": "fraud-detector", "version": "v1"},
            )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("fraud-detector", deployment_orchestrator.active_versions)

        with patch(
            "deployment_orchestrator.requests.post",
            return_value=self.worker_result("failed"),
        ):
            response = self.client.post(
                "/deployments",
                json={"model_name": "fraud-detector", "version": "v1"},
            )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("fraud-detector", deployment_orchestrator.active_versions)

    def test_rollback_success_and_failure_preserve_active_version(self):
        deployment_orchestrator.active_versions["fraud-detector"] = "v2"
        with patch(
            "deployment_orchestrator.requests.post",
            return_value=self.worker_result(),
        ):
            response = self.client.post(
                "/models/fraud-detector/rollback", json={"version": "v1"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["previous_version"], "v2")
        self.assertEqual(
            deployment_orchestrator.active_versions["fraud-detector"], "v1"
        )

        deployment_orchestrator.active_versions["fraud-detector"] = "v2"
        with patch(
            "deployment_orchestrator.requests.post",
            side_effect=requests.Timeout("timed out"),
        ):
            response = self.client.post(
                "/models/fraud-detector/rollback", json={"version": "v1"}
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            deployment_orchestrator.active_versions["fraud-detector"], "v2"
        )

    def test_rollback_validation_and_malformed_worker_result(self):
        self.assertEqual(
            self.client.post(
                "/models/fraud-detector/rollback", json={"version": "v1"}
            ).status_code,
            409,
        )
        deployment_orchestrator.active_versions["fraud-detector"] = "v2"
        self.assertEqual(
            self.client.post(
                "/models/fraud-detector/rollback", json={"version": "v2"}
            ).status_code,
            409,
        )
        self.assertEqual(
            self.client.post(
                "/models/fraud-detector/rollback", json={"version": "missing"}
            ).status_code,
            404,
        )

        malformed = Mock()
        malformed.raise_for_status.return_value = None
        malformed.json.return_value = ["not", "an", "object"]
        with patch("deployment_orchestrator.requests.post", return_value=malformed):
            response = self.client.post(
                "/deployments",
                json={"model_name": "fraud-detector", "version": "v1"},
            )
        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            deployment_orchestrator.active_versions["fraud-detector"], "v2"
        )


if __name__ == "__main__":
    unittest.main()
