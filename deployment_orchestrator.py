 
import os

from flask import Flask, jsonify, request
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

WORKER_URL = os.getenv("DEPLOYMENT_WORKER_URL", "http://127.0.0.1:5001").rstrip("/")

# Prototype-only in-memory state.
models = {}
active_versions = {}
last_known_good = {}
deployment_history = []
alert_history = []
promotion_paused = set()


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "deployment-orchestrator",
    })


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
            "error": (
                "model_name, version, and artifact_path are required"
            )
        }), 400

    versions = models.setdefault(model_name, {})

    if version in versions:
        return jsonify({
            "error": "This model version is already registered"
        }), 409

    versions[version] = {
        "version": version,
        "artifact_path": artifact_path,
    }

    return jsonify({
        "message": "Model version registered",
        "model_name": model_name,
        **versions[version],
    }), 201


@app.get("/models")
def list_models():
    return jsonify(models)


def _send_deployment(model_name, version):
    model = models.get(model_name, {}).get(version)

    if model is None:
        return None, (
            jsonify({
                "error": "Model or version is not registered"
            }),
            404,
        )

    try:
        response = requests.post(
            f"{WORKER_URL}/deploy",
            json={
                "model_name": model_name,
                "version": version,
                "artifact_path": model["artifact_path"],
            },
            timeout=10,
        )

        response.raise_for_status()
        result = response.json()

    except (requests.RequestException, ValueError):
        return None, (
            jsonify({
                "error": "Deployment worker is unavailable or failed"
            }),
            502,
        )

    if not isinstance(result, dict) or result.get("status") != "succeeded":
        return jsonify({
            "error": "Deployment did not succeed",
            "worker_result": result
        }), 502

    active_versions[model_name] = version
    deployment_history.append(result)

    return result, None


@app.post("/deployments")
def create_deployment():
    data = request.get_json(silent=True) or {}

    model_name = data.get("model_name")
    version = data.get("version")

    if not isinstance(model_name, str) or not isinstance(version, str):
        return jsonify({
            "error": "model_name and version are required"
        }), 400

    if model_name in promotion_paused:
        return jsonify({
            "error": (
                "Promotion is paused for this model "
                "due to an unresolved alert"
            )
        }), 423

    result, error = _send_deployment(model_name, version)

    if error:
        return error

    return jsonify({
        "message": "Deployment completed (worker simulation)",
        "active_version": version,
        "deployment": result,
    }), 201


@app.get("/deployments")
def list_deployments():
    return jsonify({
        "active_versions": active_versions,
        "last_known_good": last_known_good,
        "promotion_paused": sorted(promotion_paused),
        "history": deployment_history,
    })


@app.post("/models/<model_name>/rollback")
def rollback_model(model_name):
    if model_name not in active_versions:
        return jsonify({
            "error": "No active version to roll back from"
        }), 409

    data = request.get_json(silent=True) or {}

    target = (
        data.get("version")
        or last_known_good.get(model_name)
    )

    if not isinstance(target, str) or not target.strip():
        return jsonify({
            "error": (
                "Provide a target version or configure "
                "a last-known-good version"
            )
        }), 400

    if target == active_versions.get(model_name):
        return jsonify({
            "error": "That version is already active"
        }), 409

    result, error = _send_deployment(model_name, target)

    if error:
        return error

    return jsonify({
        "message": "Rollback completed (worker simulation)",
        "active_version": target,
        "deployment": result,
    }), 201


@app.post("/alerts")
def receive_alert():
    """Receive an alert and apply deterministic deployment safeguards."""

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "request body must be a JSON object"
        }), 400

    model_name = data.get("model_name")
    severity = data.get("severity")
    alert_type = data.get("alert_type")

    if not isinstance(model_name, str) or not model_name.strip():
        return jsonify({
            "error": "model_name is required"
        }), 400

    if severity not in {"info", "warning", "critical"}:
        return jsonify({
            "error": "severity must be info, warning, or critical"
        }), 400

    if not isinstance(result, dict) or result.get("status") != "succeeded":
        return jsonify({
            "error": "alert_type is required"
        }), 400

    record = {
        **data,
        "received_by_orchestrator": True,
        "pipeline_action": "none",
    }

    # Critical alerts pause new promotions.
    # Automatic rollback is deliberately not enabled.
    if severity == "critical":
        promotion_paused.add(model_name)
        record["pipeline_action"] = "promotion_paused"

    elif severity == "warning":
        record["pipeline_action"] = "validation_recommended"

    alert_history.append(record)

    return jsonify({
        "message": "Alert received",
        "alert": record,
    }), 202


@app.post("/models/<model_name>/resume-promotion")
def resume_promotion(model_name):
    data = request.get_json(silent=True) or {}

    if data.get("confirmed") is not True:
        return jsonify({
            "error": (
                "Set confirmed=true after reviewing the alert"
            )
        }), 400

    promotion_paused.discard(model_name)

    return jsonify({
        "message": "Promotion resumed",
        "model_name": model_name,
    }), 200


@app.get("/alerts")
def list_alerts():
    return jsonify({"alerts": alert_history})


if __name__ == "__main__":
    app.run(
        host=os.getenv("DEPLOYMENT_HOST", "127.0.0.1"),
        port=int(os.getenv("DEPLOYMENT_PORT", "5002")),
        debug=False
    )
