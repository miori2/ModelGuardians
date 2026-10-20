# ModelGuardians

ModelGuardians is a local prototype for monitoring model signals, requesting
Ollama-generated analysis, recording hyperparameter tuning candidates and
metrics, displaying those records in a dashboard, and simulating model
deployment orchestration. It does not train models or deploy artifacts to a
production runtime.

## 👥 Team Members & Contributions
As per course requirements, all three members must actively collaborate and commit to this repository.
*   **[Michael Orishagbemi] ([miori2])** - Role/Focus (e.g., Data Preprocessing, Feature Engineering)
*   **[Christian Douglass] ([GitHub Username])** - Role/Focus (e.g., Model Architecture, Training Pipelines)
*   **[Abdul Quayyum Yussuf] ([GitHub Username])** - Role/Focus (e.g., Hyperparameter Tuning, Evaluation & Deployment)

---

## 🚀 Live Demo
*   **Application URL:** [ModelGuardians dashboard](https://silver-space-train-v55jwj49jjq26rj-5000.app.github.dev/dashboard)
*   *Note: This live demo must be fully functional and accessible during the live project presentation.*

### 🎥 Project Walkthrough
[Project demo walkthrough](https://drive.google.com/file/d/1dycUKW-mhL9nzp2h5onYjDW5MginaetI/view)
*(Ensure your `demo.gif` is added to the repository and correctly linked above before the final submission deadline)*

---

## Requirements and installation

- Python 3.10 or newer (the source uses Python 3.10 union-type syntax; the
  dependency set also includes Flask 3.1+).
- Git and pip.
- Ollama is optional for unit tests and for statistical monitoring/tuning
  fallback behavior. Install it separately only if you want live LLM analysis
  or Ollama-generated tuning candidates.

```bash
git clone https://github.com/miori2/ModelGuardians.git
cd ModelGuardians
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The runtime stores monitoring sessions and tuning jobs in process memory.
Restarting the service clears this data. The Plotly JavaScript chart library is
loaded from its CDN by the dashboard; charts need browser access to that CDN.

## Tests

Run the complete standard-library `unittest` suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic inputs and Flask test clients. Ollama HTTP calls and
deployment-worker calls are mocked; no Ollama server, model download, trained
model, or network service is needed. The deployment worker's success behavior
is exercised locally and is explicitly a simulation.

## Run the services

Run each service in a separate terminal after activating the virtual
environment:

```bash
# Optional: install/start Ollama separately, then pull the configured model.
ollama serve
ollama pull llama3.2

# Main monitoring, tuning, and dashboard API (default 127.0.0.1:5000).
python ModelGuardians.py

# Deployment worker simulation (default 127.0.0.1:5001).
python deployment_worker.py

# Deployment orchestrator (default 127.0.0.1:5002).
python deployment_orchestrator.py
```

The monitoring service can run without Ollama: event ingestion remains
available and returns a fallback diagnosis when Ollama transport fails.
Tuning falls back to deterministic candidates when generation fails or yields
invalid suggestions. Set these variables before launching the corresponding
processes:

| Variable | Default | Service |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Monitoring/tuning/dashboard Ollama client |
| `OLLAMA_MODEL` | `llama3.2` | Monitoring/tuning/dashboard Ollama client |
| `FLASK_HOST` | `127.0.0.1` | Main app bind address |
| `PORT` | `5000` | Main app port |
| `DEPLOYMENT_WORKER_HOST` | `127.0.0.1` | Worker bind address |
| `DEPLOYMENT_WORKER_PORT` | `5001` | Worker port |
| `DEPLOYMENT_HOST` | `127.0.0.1` | Orchestrator bind address |
| `DEPLOYMENT_PORT` | `5002` | Orchestrator port |
| `DEPLOYMENT_WORKER_URL` | `http://127.0.0.1:5001` | Orchestrator's worker URL |

For local use, keep the worker URL aligned with the worker's host and port.
These Flask development servers are not production-hardened.

## Monitoring API

The examples below use the main service at `http://127.0.0.1:5000`.

Create a session. A baseline is optional; when supplied it requires a finite
`mean`, and optional finite, non-negative `stddev`:

```bash
curl -X POST http://127.0.0.1:5000/monitoring/sessions \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-detector","baseline":{"mean":0.5,"stddev":0.1}}'
```

Replace the session/job/deployment ID placeholders in these examples with IDs
returned by the corresponding create/request response.

Post an event using the returned session ID. If `features` is present, it is
used for signal calculations; otherwise a numeric `prediction` is used.
Non-finite numeric values are rejected with HTTP 400. Other/missing numeric
signals produce no observed mean:

```bash
curl -X POST http://127.0.0.1:5000/monitoring/sessions/<SESSION_ID>/events \
  -H 'Content-Type: application/json' \
  -d '{"features":[0.92],"prediction":1}'
```

Get session metadata at
`GET /monitoring/sessions/<SESSION_ID>`. Subscribe to new enriched events at
`GET /monitoring/sessions/<SESSION_ID>/stream` using Server-Sent Events. The
statistical signals are simple baseline comparisons, not a trained detector.
The optional Ollama diagnosis is generated from each event; it is advisory,
not a model correction.

## Hyperparameter tuning API

Create a job by specifying a model label, metric, direction, trial count, and
search space. Domains are categorical lists, typed categorical objects, or
inclusive integer/float bounds:

```bash
curl -X POST http://127.0.0.1:5000/tuning/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "model_name":"fraud-detector",
    "metric_name":"validation_auc",
    "direction":"maximize",
    "max_trials":4,
    "search_space":{
      "n_estimators":{"type":"int","min":50,"max":200},
      "max_depth":[3,5,8],
      "learning_rate":{"type":"float","min":0.01,"max":0.2}
    }
  }'
```

`max_trials` defaults to 10 and must be between 1 and 100. Search spaces allow
up to 20 parameters and 100 categorical values per parameter. The service
returns exactly the requested number of **distinct** candidate configurations
or rejects the job with HTTP 400 if the space cannot supply that count. Ollama
suggestions are validated against the declared domains; duplicate, malformed,
or out-of-domain suggestions are discarded and deterministic in-domain
candidates fill remaining slots. Each candidate identifies its source as
`ollama` or `fallback`.

Submit a measured metric once for each candidate after evaluating it yourself;
the API does not train or evaluate a model:

```bash
curl -X POST http://127.0.0.1:5000/tuning/jobs/<JOB_ID>/trials \
  -H 'Content-Type: application/json' \
  -d '{"candidate_id":"candidate-1","metric_value":0.93}'
curl http://127.0.0.1:5000/tuning/jobs/<JOB_ID>
```

The returned job includes submitted trials and the current `best_trial`.
Invalid request bodies and values return HTTP 400; unknown jobs/candidates
return HTTP 404; a duplicate candidate submission returns HTTP 409. Jobs and
results are in memory and are cleared on restart.

## Dashboard

Open <http://127.0.0.1:5000/dashboard>. It refreshes the monitoring/tuning
charts every 15 seconds, supports a monitoring-session filter, and has a manual
refresh button. The dashboard summarizes active alerts and predictive
warnings, lists recent events, and reports the loaded fraud model. The API data
is available at `GET /api/dashboard`, optionally with
`?session_id=<SESSION_ID>`. Monitoring metrics and forecasts follow that
session filter; tuning results are global to the in-memory tuning store.

Dashboard data is available as JSON from `GET /api/dashboard`, optionally
filtered with `?session_id=<SESSION_ID>`. Request LLM analysis with
`POST /api/dashboard/insights` and an optional JSON `session_id`.

## Credit-card fraud dataset workflow

The project includes a workflow for the public Kaggle
**Credit Card Fraud Detection Dataset 2023**. The first training run downloads
the CSV into `data/raw/` (about 143 MB), inspects its columns, trains a
class-weighted histogram gradient-boosting classifier, saves it under
`artifacts/`, and prints held-out evaluation metrics:

```bash
python fraud_pipeline.py train
```

The dataset has 568,630 rows, 28 anonymized `V1`–`V28` features, `Amount`, an
`id` identifier (excluded from training), and the `Class` fraud label. Its
classes are balanced in this dataset; precision and probability calibration
may therefore differ from real-world fraud prevalence. The training command
does a stratified train/test split and reports precision, recall, F1, ROC-AUC,
average precision, and a confusion matrix. Dataset and model files are local
and git-ignored; they are not checked into the repository.

After training, start or restart the app so it loads the saved model:

```bash
python ModelGuardians.py
```

Replay up to 500 sampled dataset rows through the trained model and into a
monitoring session (the app must be running):

```bash
python fraud_pipeline.py replay --limit 500
```

Open `http://127.0.0.1:5000/dashboard`. The **Observed signal and short-term
forecast** chart plots each row's model-predicted fraud probability; the
forecast is a linear extrapolation of recent probabilities, not a second
model prediction. Batch ingestion is capped at 1,000 rows per request and
skips per-event LLM calls by default. To adjust the replay pace, use
`--batch-size` and `--delay`.

Run measured hyperparameter trials and submit their validation average
precision to the existing tuning API so they appear in **Hyperparameter tuning
results**:

```bash
python fraud_pipeline.py tune --max-trials 3
```

The app must be running for both `replay` and `tune`. The selected metric is
average precision, useful for comparing fraud classifiers; tuning jobs remain
in memory and disappear when the app restarts. The training, replay, and
tuning commands accept `--data path/to/file.csv` to use a local CSV instead of
downloading the Kaggle copy. Set `FRAUD_MODEL_PATH` if the model artifact is
stored at a non-default path. The prediction API also accepts individual
feature rows at
`POST /monitoring/sessions/<SESSION_ID>/predict`; the dataset replay command
uses the batch variant.

The short-term forecast is a simple linear extrapolation of recorded event
means (up to five points), not a trained forecasting model or a guarantee.
Click **Generate insights** to send aggregated metrics and recent events to
Ollama through `POST /api/dashboard/insights`; a selected session can be
provided as JSON `session_id`. If Ollama is unavailable, the endpoint responds
with HTTP 503 and the dashboard displays the error. Dashboard charts/data do
not require Ollama.

## Deployment orchestration prototype

The worker accepts deployment requests and records their metadata in memory;
it does not load, validate, serve, or activate a model artifact. Its `succeeded`
status means only that the request was recorded by the simulation. The
orchestrator tracks registered versions and active-version labels in its own
in-memory process; it is not integrated with the monitoring/tuning app and
does not provide production deployment, persistence, authentication, or
artifact storage.

Register versions and request a simulated deployment at
`http://127.0.0.1:5002`:

```bash
curl -X POST http://127.0.0.1:5002/models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-detector","version":"v1","artifact_path":"/models/fraud-v1"}'
curl -X POST http://127.0.0.1:5002/deployments \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-detector","version":"v1"}'
curl http://127.0.0.1:5002/deployments
```

The worker status endpoint is
`GET http://127.0.0.1:5001/deployments/<DEPLOYMENT_ID>`. To simulate rollback,
register and deploy a second version, then request the first version:

```bash
curl -X POST http://127.0.0.1:5002/models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-detector","version":"v2","artifact_path":"/models/fraud-v2"}'
curl -X POST http://127.0.0.1:5002/deployments \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-detector","version":"v2"}'
curl -X POST http://127.0.0.1:5002/models/fraud-detector/rollback \
  -H 'Content-Type: application/json' \
  -d '{"version":"v1"}'
```

An unavailable worker or failed worker result returns HTTP 502 and does not
change the orchestrator's active version. Registries and history are lost when
either service restarts.

## Troubleshooting

- **Ollama connection errors:** check that Ollama is running at `OLLAMA_HOST`
  and that `OLLAMA_MODEL` has been pulled. Monitoring still accepts valid
  events; tuning marks generated candidates as fallback when needed.
- **Deployment worker unavailable:** start `deployment_worker.py` before
  calling the orchestrator, or align `DEPLOYMENT_WORKER_URL` with the worker
  host/port.
- **Port already in use:** change `PORT`, `DEPLOYMENT_WORKER_PORT`, or
  `DEPLOYMENT_PORT`; update `DEPLOYMENT_WORKER_URL` to match the worker.
- **Tuning job returns HTTP 400 for `max_trials`:** lower the count or expand
  the search space so it can provide that many distinct configurations.
- **Dashboard charts unavailable:** confirm browser access to the Plotly CDN;
  the JSON dashboard endpoints remain available independently.
