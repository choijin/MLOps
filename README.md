# NYC Taxi Fare Prediction MLOps

This project builds a small local MLOps workflow for predicting NYC green taxi
fares. The goal was to move beyond a notebook-only model and practice the pieces
around it: training pipelines, experiment tracking, model registration,
containerization, and a prediction API.

The project currently runs locally. MLflow is used for tracking and model
artifacts, and FastAPI serves predictions through a local web service.

## What It Does

- trains a Ridge regression fare prediction model
- logs parameters, metrics, and model artifacts to MLflow
- compares candidate and champion models
- serves predictions with FastAPI
- supports Docker Compose for local MLflow, training, and API services
- includes a bundled model in the API image so the app can be tried without
  training first

## Project Layout

- `01_initial_ml_build/`: initial notebook exploration
- `02_model_training/`: training pipeline, feature code, model comparison, Airflow DAG
- `03_deployment/`: FastAPI app, Dockerfile, and bundled demo model
- `docker-compose.yml`: local MLflow, training, and API services

## Quick Try

The API image includes a bundled trained model for demo use. This means someone
can run the web service and send a prediction request without setting up MLflow
or training a model first.

```bash
docker compose up --build api
```

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

Response shape:

```json
{
  "predictions": [14.936]
}
```

The exact value can change when a different model is served.

## Train And Serve Through MLflow

For the fuller MLOps workflow, start MLflow and run the training pipeline:

```bash
docker compose up --build mlflow
docker compose run --rm train
```

Training writes champion model metadata to:

```text
02_model_training/artifacts/model_result.json
```

Copy the champion `run_id` into a local deployment env file:

```bash
cp 03_deployment/config/deployment.env.example 03_deployment/config/deployment.env
```

```env
RUN_ID=<champion_run_id>
MODEL_ARTIFACT_PATH=final_model
```

Then start the API:

```bash
docker compose up --build api
```

When `RUN_ID` is set, the API loads `runs:/<RUN_ID>/final_model` from MLflow.
When `RUN_ID` is not set, it falls back to the bundled model in
`03_deployment/saved_models/model.joblib`.

## Local Commands Without Docker

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

- This is a local-first project, not a production cloud deployment.
- `deployment.env`, MLflow files, databases, caches, and training artifacts are
  intentionally ignored by Git.
- The bundled model is included mainly so the Docker image can be pulled and
  tested easily. The MLflow path is the main training and model-selection flow.
