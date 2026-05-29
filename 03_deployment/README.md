# 03 Deployment

This stage contains the deployment-focused version of the project.

Layout:

- `app/`: FastAPI application modules
- `src/data/`: config loading helpers used at inference time
- `src/features/`: feature engineering and preprocessing helpers
- `src/models/`: prediction logic and MLflow artifact resolution
- `config/`: model and build configuration files

The deployment service loads a trained model from the mounted MLflow artifact store.
Provide `RUN_ID` plus the MLflow tracking URI so the API can load
`runs:/<run_id>/final_model` at startup.

For local Docker usage, the root `docker-compose.yml` starts:

- an `mlflow` service backed by `./mlflow`
- an `api` service that reads `03_deployment/config/deployment.env` for the current `RUN_ID`

`02_model_training/compare_models.py` writes `03_deployment/config/deployment_config.json`
with the active champion `run_id`, and you can copy that value into
`03_deployment/config/deployment.env` when you want deployment to serve that run.
