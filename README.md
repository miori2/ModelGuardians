# ModelGuardians
Machine Learning tool that is able to analyze when a model is working incorrectly and utilizes an LLM to diagnose the problem and provide tools to improve and redeploy the model.

*   **[Michael Orishagbemi] ([miori2])** - Role/Focus (e.g., Data Preprocessing, Feature Engineering)
*   **[Christian Douglass] ([GitHub Username])** - Role/Focus (e.g., Model Architecture, Training Pipelines)
*   **[Abdulquayyum Yussuf] ([AYO-YUSSUF])** - Role/Focus (e.g., Hyperparameter Tuning, Evaluation & Deployment)


## 🚀 Live Demo
*   **Application URL:** [Insert Link to Live App, e.g., Hugging Face Spaces / Streamlit Sharing / Vercel]
*   *Note: This live demo must be fully functional and accessible during the live project presentation.*

### 🎥 Project Walkthrough
![Project Demo Walkthrough](path/to/your/demo.gif)
*(Ensure your `demo.gif` is added to the repository and correctly linked above before the final submission deadline)*

## 🛠️ Installation & Setup

Follow these steps to set up the environment and run the project locally.

### Prerequisites
*   Python 3.8+
*   `pip` or `conda`
*   Git

### Local Installation
```bash
# Clone the repository
git clone https://github.com/username/repository-name.git
cd repository-name

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

## Real-time monitoring API

Start Ollama and pull a model (the default is `llama3.2`), then run:

```bash
ollama serve
ollama pull llama3.2
python ModelGuardians.py
```

The implementation is split across [app.py](app.py), [monitoring.py](monitoring.py),
and [ollama_client.py](ollama_client.py). `ModelGuardians.py` remains as a
backward-compatible launcher.

Create a monitoring session with an optional baseline mean and standard
deviation:

```bash
curl -X POST http://127.0.0.1:5000/monitoring/sessions \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-detector","baseline":{"mean":0.5,"stddev":0.1}}'
```

Send each prediction or feature vector as it arrives. The response contains
statistical anomaly/drift signals and the Ollama diagnosis:

```bash
curl -X POST http://127.0.0.1:5000/monitoring/sessions/<SESSION_ID>/events \
  -H 'Content-Type: application/json' \
  -d '{"features":[0.92],"prediction":1}'
```

Clients can subscribe to `/monitoring/sessions/<SESSION_ID>/stream` using
Server-Sent Events to receive every enriched event in real time. Configure
`OLLAMA_HOST`, `OLLAMA_MODEL`, `FLASK_HOST`, and `PORT` through environment
variables. If Ollama is unavailable, ingestion remains available and clearly
reports that the LLM diagnosis could not be generated.

## Automated hyperparameter tuning API

Tuning jobs use the same Ollama configuration as monitoring. A job declares a
search space, the metric to track, and whether that metric should be minimized
or maximized. Search-space values can be categorical lists, typed categorical
domains, or inclusive numeric ranges:

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

`POST /tuning/jobs` returns the job, candidate configs, and a
`candidate_id` for each config. `max_trials` is optional (default `10`, maximum
`100`); each parameter must have a non-empty domain, with no more than 20
parameters or 100 categorical values per parameter. Categorical domains may
also be written as `{"type":"categorical","values":[...]}`. Ollama suggestions
are strictly checked against the declared search space; if Ollama is
unavailable or its output is malformed or out of range, safe deterministic
candidates are used instead. Each candidate reports its `source` as
`ollama` or `fallback`.

Submit a measured metric for a candidate after evaluating that configuration.
Each candidate can be submitted once:

```bash
curl -X POST http://127.0.0.1:5000/tuning/jobs/<JOB_ID>/trials \
  -H 'Content-Type: application/json' \
  -d '{"candidate_id":"candidate-1","metric_value":0.93}'
```

The response includes all submitted trials and `best_trial`, selected according
to the job's metric direction. Retrieve current job state at
`GET /tuning/jobs/<JOB_ID>`. Jobs and trial results are held in memory, so they
are cleared when the application restarts. Invalid job/trial requests return
HTTP 400, unknown jobs or candidates return HTTP 404, and duplicate trial
submissions return HTTP 409.

## Visualization dashboard

Open `http://127.0.0.1:5000/dashboard` while the application is running to view
interactive Plotly charts for observed monitoring signals, a short-term linear
trend forecast, and submitted tuning metrics. The dashboard can be filtered to
one monitoring session, refreshes data every 15 seconds, and can be refreshed
on demand. Plotly.js is loaded from its CDN. Forecasts are indicative linear
projections, not guarantees. Click **Generate insights** to request an LLM
analysis of aggregated metrics through the configured Ollama model; if Ollama
is unavailable, the dashboard reports the issue without interrupting
monitoring or chart data.

Dashboard data is available as JSON from `GET /api/dashboard`, optionally
filtered with `?session_id=<SESSION_ID>`. Request LLM analysis with
`POST /api/dashboard/insights` and an optional JSON `session_id`.
