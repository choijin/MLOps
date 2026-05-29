# MLOps

End-to-end NYC taxi fare modeling project with local MLflow tracking, Airflow
orchestration, model training, and a FastAPI prediction service.

## Project Layout

- `01_initial_ml_build/`: exploratory notebook work and initial modeling
- `02_model_training/`: reusable training pipeline, Airflow DAG, feature code, and
  MLflow model registration
- `03_deployment/`: FastAPI app, Docker setup, and inference-time feature logic
- `mlflow/`: local MLflow backend and artifact store, ignored by Git

## Start MLflow Locally

Run MLflow from the repo root:

```bash
uv run mlflow ui \
  --backend-store-uri sqlite:////Users/jinchoi/Documents/Python/mlops/mlflow/db/mlflow.db \
  --default-artifact-root /Users/jinchoi/Documents/Python/mlops/mlflow \
  --host 127.0.0.1 \
  --port 5001
```

Open the UI at `http://127.0.0.1:5001`.

Notes:

- This command creates the MLflow SQLite database at the path above.
- If port `5001` is unavailable, choose another port and use the same port in
  training/deployment commands.
- Training and prediction scripts should use the HTTP tracking URI, for example
  `http://127.0.0.1:5001`, not the `.db` file path.
- The scripts talk to the MLflow server. The server talks to the database.

## Train Model With MLflow Tracking

In a second terminal, run:

```bash
cd /Users/jinchoi/Documents/Python/mlops/02_model_training

uv run python pipeline.py \
  --train-start-year 2024 \
  --train-year 2025 \
  --train-end-month 1 \
  --spec-path config/model_build_spec.json \
  --tracking-uri http://127.0.0.1:5001 \
  --experiment-name zoomcamp-model \
  --registered-model-name nyc-taxi-ridge
```

The pipeline:

- loads the model spec from `config/model_build_spec.json`
- builds train/validation/test splits
- trains a Ridge model
- logs params, metrics, feature spec, and model artifacts to MLflow
- registers the model as `nyc-taxi-ridge`
- compares candidate and champion models for promotion

After training, copy the printed `Run ID` when you want to serve that exact run.
MLflow organizes runs under experiment folders automatically.

## Airflow Setup

From `02_model_training/`:

```bash
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
```

Initialize Airflow:

```bash
airflow db migrate
```

Inspect DAGs:

```bash
airflow dags list
airflow dags report
airflow dags reserialize
```

Test a DAG:

```bash
airflow dags test example_dag 2026-04-01
```

## Containerize MLflow

From the repo root:

```bash
docker compose up --build mlflow
```

Then train a model against the containerized MLflow service:

```bash
cd /Users/jinchoi/Documents/Python/mlops/02_model_training

uv run python pipeline.py \
  --train-start-year 2024 \
  --train-year 2025 \
  --train-end-month 1 \
  --spec-path config/model_build_spec.json \
  --tracking-uri http://127.0.0.1:5001 \
  --experiment-name zoomcamp-model \
  --registered-model-name nyc-taxi-ridge
```

Copy the resulting `run_id`.

## Containerize FastAPI

Serve a trained run through the API by adding the run ID to the deployment env
file:

```bash
cd /Users/jinchoi/Documents/Python/mlops

cat > 03_deployment/config/deployment.env <<'EOF'
RUN_ID=<your_run_id>
MODEL_ARTIFACT_PATH=final_model
EOF
```

Then start the API:

```bash
docker compose up --build api
```

The Docker API service uses `MLFLOW_TRACKING_URI=http://mlflow:5001` inside the
compose network and mounts the local `./mlflow` artifact store.

## Test The API

Health check:

```bash
curl http://127.0.0.1:9696/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "VendorID": 2.0,
        "lpep_pickup_datetime": "2025-01-15T08:30:00",
        "lpep_dropoff_datetime": "2025-01-15T08:48:00",
        "passenger_count": 1.0,
        "trip_distance": 2.7,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": 74,
        "DOLocationID": 41,
        "trip_type": 1.0
      }
    ]
  }'
```

Expected response shape:

```json
{
  "predictions": [18.742]
}
```

The numeric value will vary with the trained model and run ID.

## Run Prediction Without Docker

From `03_deployment/`:

```bash
cd /Users/jinchoi/Documents/Python/mlops/03_deployment

RUN_ID=<your_run_id> \
MLFLOW_ARTIFACT_ROOT=/Users/jinchoi/Documents/Python/mlops/mlflow \
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9696
```

If your MLflow server is not running at `http://127.0.0.1:5001`, set
`MLFLOW_TRACKING_URI` too:

```bash
RUN_ID=<your_run_id> \
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9696
```

## Function Use Cases

### Train from Python

Use `run_train` when another script already built the split data and loaded the
model spec:

```python
from pathlib import Path

from data.dataset_build import load_spec, load_split_data
from models.train import run_train

spec_path = Path("config/model_build_spec.json")
spec = load_spec(spec_path)
feature_cols = spec["num_cols"] + spec["ohe_cols"] + spec["te_cols"]

data = load_split_data(
    train_start_year=2024,
    train_year=2025,
    train_end_month=1,
    feature_cols=feature_cols,
    ohe_cols=spec["ohe_cols"],
    target=spec["target"],
)

run_train(
    data=data,
    spec=spec,
    spec_path=spec_path,
    tracking_uri="http://127.0.0.1:5001",
    experiment_name="zoomcamp-model",
    registered_model_name="nyc-taxi-ridge",
    candidate_alias="candidate",
    export_dir=Path("artifacts/model_exports/candidate/manual"),
)
```

### Predict from Python

Use `predict_records` for batch scoring without starting FastAPI:

```python
from src.models.predict import predict_records

records = [
    {
        "VendorID": 2.0,
        "lpep_pickup_datetime": "2025-01-15T08:30:00",
        "lpep_dropoff_datetime": "2025-01-15T08:48:00",
        "passenger_count": 1.0,
        "trip_distance": 2.7,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": 74,
        "DOLocationID": 41,
        "trip_type": 1.0,
    }
]

predictions = predict_records(
    records=records,
    run_id="<your_run_id>",
    tracking_uri="http://127.0.0.1:5001",
)
print(predictions)
```

### Reuse A Loaded Model In The API

`PredictionService` loads the MLflow model once, caches it, and reuses it across
requests:

```python
from app.service import PredictionService

service = PredictionService(
    run_id="<your_run_id>",
    tracking_uri="http://127.0.0.1:5001",
)

predictions = service.predict(records)
```

## Git Notes

Runtime artifacts are intentionally ignored:

- `.venv/`
- `mlflow/`
- `*.db`
- `02_model_training/artifacts/`
- `03_deployment/config/deployment.env`
- Python cache and notebook checkpoint directories

Commit source code, specs, Docker files, docs, and tests. Keep model artifacts,
local MLflow runs, SQLite databases, and local env files out of Git.
