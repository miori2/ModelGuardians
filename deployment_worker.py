# Deployment worker receives deployment requests from the orchestrator. Right now it is able to record the request
# and simulates a successful deployment.


import os
from flask import Flask, jsonify, request
import uuid

app = Flask(__name__)

# Temporary storage for deployment jobs
deployments = {}

# Sends a status: healthy message when a GET request to health is successful
@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


# Receive a deployment request by creating an endpoint, is then able to retrive the desired model
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
            "error": "model_name, version, and artifact_path are required"
        }), 400

    deployment_id = str(uuid.uuid4())

    # Prototype only: no real model is deployed yet.
    deployments[deployment_id] = {
        "deployment_id": deployment_id,
        "model_name": model_name,
        "version": version,
        "artifact_path": artifact_path,
        "status": "succeeded",
        "message": "Deployment simulated successfully"
    }

    return jsonify(deployments[deployment_id]), 201

# Allows another program to look up the deployment
@app.get("/deployments/<deployment_id>")
def get_deployment(deployment_id):
    deployment = deployments.get(deployment_id)

    if deployment is None:
        return jsonify({"error": "Deployment not found"}), 404

    return jsonify(deployment)

# Activates the worker (type python deployment_worker.py in terminal) 
if __name__ == "__main__":
    app.run(
        host=os.getenv("DEPLOYMENT_WORKER_HOST", "127.0.0.1"),
        port=int(os.getenv("DEPLOYMENT_WORKER_PORT", "5001")),
        debug=False
    )
