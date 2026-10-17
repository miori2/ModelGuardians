
"""Prototype deployment worker. No real model is loaded."""

from flask import Flask, jsonify, request
import os
import uuid

app = Flask(__name__)

deployments = {}


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "deployment-worker",
    })


@app.post("/deploy")
def deploy_model():
    data = request.get_json(silent=True) or {}

    model_name = data.get("model_name")
    version = data.get("version")
    artifact_path = data.get("artifact_path")

    if not all(
        isinstance(value, str) and value.strip()
        for value in (model_name, version, artifact_path)
    ):
        return jsonify({
            "error": (
                "model_name, version, and artifact_path are required"
            )
        }), 400

    deployment_id = str(uuid.uuid4())

    deployments[deployment_id] = {
        "deployment_id": deployment_id,
        "model_name": model_name,
        "version": version,
        "artifact_path": artifact_path,
        "status": "succeeded",
        "message": (
            "Deployment simulated successfully; "
            "no real model was loaded."
        ),
    }

    return jsonify(deployments[deployment_id]), 201


@app.get("/deployments/<deployment_id>")
def get_deployment(deployment_id):
    deployment = deployments.get(deployment_id)

    if deployment is None:
        return jsonify({
            "error": "Deployment not found"
        }), 404

    return jsonify(deployment)


if __name__ == "__main__":
    app.run(
        host=os.getenv("WORKER_HOST", "127.0.0.1"),
        port=int(os.getenv("WORKER_PORT", "5001")),
        debug=False,
    )
