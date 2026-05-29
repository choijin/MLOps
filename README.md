# MLOps NYC Taxi Fare Prediction

This project is an end-to-end MLOps workflow for training and serving a NYC taxi
fare prediction model.

It currently runs on local services:

- MLflow runs locally for experiment tracking, model registry, and artifacts.
- FastAPI runs locally as the prediction web service.
- Docker Compose can start the local MLflow and API services together.

This is not deployed to a cloud environment yet. The current setup is meant for
local development and demonstration of the MLOps workflow.

## Project Structure

- `01_initial_ml_build/`: initial notebook exploration
- `02_model_training/`: training pipeline, model comparison, and Airflow DAG
- `03_deployment/`: FastAPI prediction service and Docker files
- `docker-compose.yml`: local MLflow and API services

## Main Workflow

The usual workflow is:

1. Start MLflow locally.
2. Train a model and log it to MLflow.
3. Export or bundle the selected model for serving.
4. Send records to the API and receive fare predictions.

You do not need to run every command in this README every time. Use the section
that matches what you are trying to do.

## Run MLflow Locally

From the repository root:

```bash
mkdir -p mlflow/db

uv run mlflow server \
  --backend-store-uri sqlite:///$PWD/mlflow/db/mlflow.db \
  --default-artifact-root $PWD/mlflow \
  --host 127.0.0.1 \
  --port 5001
```

MLflow will be available at:

```text
http://127.0.0.1:5001
```

Training code should use the HTTP tracking URI, not the SQLite database path:

```text
http://127.0.0.1:5001
```

## Train A Model

In another terminal:

```bash
cd 02_model_training

uv run python pipeline.py \
  --train-start-year 2024 \
  --train-year 2025 \
  --train-end-month 1 \
  --spec-path config/model_build_spec.json \
  --tracking-uri http://127.0.0.1:5001 \
  --experiment-name zoomcamp-model \
  --registered-model-name nyc-taxi-ridge
```

The training pipeline logs parameters, metrics, and the trained model to MLflow.
It also exports the champion model under `02_model_training/artifacts/`.

For the standalone API image, copy the selected exported model into:

```text
03_deployment/saved_models/model.joblib
```

The current repository includes a bundled champion model for local demo serving.

## Run With Docker Compose

Start MLflow:

```bash
docker compose up --build mlflow
```

Start the API:

```bash
docker compose up --build api
```

The API first looks for `03_deployment/saved_models/model.joblib`. If that file
is present, the service can run without MLflow at inference time.

## Run The API Without Docker

From `03_deployment/`:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9696
```

If no bundled model is available, the API can still load from MLflow by setting
`RUN_ID` and `MLFLOW_TRACKING_URI`.

## Test The API

Health check:

```bash
curl http://127.0.0.1:9696/health
```

Prediction:

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

The API returns:

```json
{
  "predictions": [18.742]
}
```

The exact prediction value depends on the trained model.

## Useful Code Entry Points

- `02_model_training/pipeline.py`: runs the full training and model comparison workflow
- `02_model_training/models/train.py`: trains and registers a candidate model
- `02_model_training/models/compare_models.py`: compares candidate and champion models
- `03_deployment/app/main.py`: starts the FastAPI app
- `03_deployment/src/models/predict.py`: loads a bundled or MLflow model and scores records

## Airflow

An Airflow DAG is included in `02_model_training/dags/`. It is useful for
orchestrating the training workflow locally, but it is not required for manually
running the project.

## Notes

- `mlflow/`, local databases, model artifacts, caches, and env files are ignored
  by Git.
- The current implementation is local-first. A production version would need a
  remote artifact store, managed tracking server, deployed API, and secret
  management.
