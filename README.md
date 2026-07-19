# Fair Price Prediction MLOps Service

This project builds a local MLOps workflow for training, tracking, selecting, and
serving a regression model through an API.

The model currently uses NYC green taxi trip data as the example dataset, but the
main focus of the project is the MLOps system around the model: reusable training
code, MLflow experiment tracking, champion model selection, Dockerized services,
and FastAPI inference.

## What It Does

- Trains a Ridge regression model for fare prediction
- Logs parameters, metrics, and model artifacts to MLflow
- Compares candidate models and exports champion model metadata
- Serves predictions through a FastAPI web service
- Uses Docker Compose to run MLflow, training, and API services locally
- Includes a demo model inside the API image so you can test the service before running the full training workflow

## Project Layout

- `01_initial_ml_build/`: initial notebook exploration
- `02_model_training/`: training pipeline, feature code, model comparison, and Airflow DAG
- `03_deployment/`: FastAPI app, Dockerfile, config, and bundled demo model
- `docker-compose.yml`: local MLflow, training, and API service orchestration

## Recommended First Run: Try the API

The API image already includes a small trained demo model. You can start the web
service and send a prediction request without setting up MLflow or training a
new model first.

```bash
docker compose up --build api
```

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

Example response:

```json
{
  "predictions": [14.936]
}
```

The exact prediction can change when a different model is served.

## Full Docker Workflow: Train, Track, And Serve

Use this path when you want to train a different model and run the full local
MLOps workflow.

```bash
docker compose up --build mlflow
```

In a second terminal:

```bash
docker compose run --rm train
```

Training logs runs to MLflow and writes the selected champion metadata to:

```text
02_model_training/artifacts/model_result.json
```

Create the API env file and set the champion run:

```bash
cp 03_deployment/config/deployment.env.example 03_deployment/config/deployment.env
```

```env
RUN_ID=<champion_run_id>
MODEL_ARTIFACT_PATH=final_model
```

Start the API:

```bash
docker compose up --build api
```

- With `RUN_ID`, the API serves `runs:/<RUN_ID>/final_model` from MLflow.
- Without `RUN_ID`, it falls back to `03_deployment/saved_models/model.joblib`.
- To update that fallback model, copy your exported champion `model.joblib` into
`03_deployment/saved_models/` before rebuilding the API image.

## Local Workflow Without Docker

Use this path when you want to run the same pieces directly on your machine.

Start MLflow:

```bash
mkdir -p mlflow/db

uv run mlflow server \
  --backend-store-uri sqlite:///$PWD/mlflow/db/mlflow.db \
  --default-artifact-root $PWD/mlflow \
  --host 127.0.0.1 \
  --port 5001
```

Train:

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

Run the API:

```bash
cd 03_deployment

uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 9696
```

## Notes

- This is a local-first MLOps project, not a production cloud deployment.
- `deployment.env`, MLflow files, databases, caches, and generated training
  artifacts are intentionally ignored by Git.
