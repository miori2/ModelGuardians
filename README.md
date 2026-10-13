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
