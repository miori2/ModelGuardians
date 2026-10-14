 
from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

WORKER_URL = "http://127.0.0.1:5001"

# Temporary in-memory registries
models = {}
active_versions = {}
deployment_history = []

# Sends a status: healthy message when a GET request to health is successful
@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

# Creates API Endpoint for registering a model
@app.post("/models")
def register_model():
    data = request.get_json(silent=True) or {}

    model_name = data.get("model_name")
    version = data.get("version")
    artifact_path = data.get("artifact_path")

    if not all(
        isinstance(value, str) and value.strip()
        for value in (model_name, version, artifact_path)
    ):
        return jsonify({
            "error": "model_name, version, and artifact_path are required"
        }), 400

    model_versions = models.setdefault(model_name, {})

    if version in model_versions:
        return jsonify({
            "error": "This model version is already registered"
        }), 409

    model_versions[version] = {
        "version": version,
        "artifact_path": artifact_path
    }

    return jsonify({
        "message": "Model version registered",
        "model_name": model_name,
        **model_versions[version]
    }), 201


# Provides a list of every currently registered model and version
@app.get("/models")
def list_models():
    return jsonify(models)

# Endpoint receives a request to deploy a model
@app.post("/deployments")
def create_deployment():
    data = request.get_json(silent=True) or {}

    model_name = data.get("model_name")
    version = data.get("version")

    if not isinstance(model_name, str) or not isinstance(version, str):
        return jsonify({
            "error": "model_name and version are required"
        }), 400

    model_versions = models.get(model_name, {})
    model = model_versions.get(version)

    if model is None:
        return jsonify({
            "error": "Model or version is not registered"
        }), 404

    try:
        response = requests.post(
            f"{WORKER_URL}/deploy",
            json={
                "model_name": model_name,
                "version": version,
                "artifact_path": model["artifact_path"]
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

    except requests.RequestException:
        return jsonify({
            "error": "Deployment worker is unavailable or failed"
        }), 502

    if result.get("status") != "succeeded":
        return jsonify({
            "error": "Deployment did not succeed",
            "worker_result": result
        }), 502

    active_versions[model_name] = version
    deployment_history.append(result)

    return jsonify({
        "message": "Deployment completed",
        "active_version": version,
        "deployment": result
    }), 201


# View deployment history
@app.get("/deployments")
def list_deployments():
    return jsonify({
        "active_versions": active_versions,
        "history": deployment_history
    })

# Allows use of a previous version of the model
@app.post("/models/<model_name>/rollback")
def rollback_model(model_name):
    previous_version = active_versions.get(model_name)

    if previous_version is None:
        return jsonify({
            "error": "No active version to roll back from"
        }), 409

    data = request.get_json(silent=True) or {}
    target_version = data.get("version")

    if not isinstance(target_version, str) or not target_version.strip():
        return jsonify({
            "error": "Provide the target version in the request"
        }), 400

    if target_version == previous_version:
        return jsonify({
            "error": "That version is already active"
        }), 409

    if target_version not in models.get(model_name, {}):
        return jsonify({
            "error": "Target version is not registered"
        }), 404

    # Rollback uses the same deployment workflow as a new deployment.
    # Save the previous version in case the request fails.
    try:
        response = requests.post(
            f"{WORKER_URL}/deploy",
            json={
                "model_name": model_name,
                "version": target_version,
                "artifact_path": models[model_name][target_version][
                    "artifact_path"
                ]
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

    except requests.RequestException:
        return jsonify({
            "error": "Rollback failed because the worker is unavailable"
        }), 502

    if result.get("status") != "succeeded":
        return jsonify({
            "error": "Rollback deployment did not succeed",
            "worker_result": result
        }), 502

    active_versions[model_name] = target_version
    deployment_history.append(result)

    return jsonify({
        "message": "Rollback completed",
        "previous_version": previous_version,
        "active_version": target_version,
        "deployment": result
    })


# Activates orchestrator when (python deployment_orchestrator.py) is run in the terminal
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
