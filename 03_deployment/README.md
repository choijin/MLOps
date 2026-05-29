# 03 Deployment

This stage contains the deployment-focused version of the project.

Layout:

- `app/`: FastAPI application modules
- `src/data/`: config loading helpers used at inference time
- `src/features/`: feature engineering and preprocessing helpers
- `src/models/`: prediction logic for bundled or MLflow models
- `config/`: model and build configuration files
- `saved_models/`: optional bundled model artifacts for standalone serving

The deployment service first looks for `saved_models/model.joblib`. If that file
exists, the API can serve predictions without contacting MLflow at inference
time.

If no bundled model is present, provide `RUN_ID` plus the MLflow tracking URI so
the API can load `runs:/<run_id>/final_model` at startup.

For local Docker usage, the root `docker-compose.yml` starts:

- an `mlflow` service backed by `./mlflow`
- an `api` service that serves the bundled model by default

`02_model_training/models/compare_models.py` exports the active champion model.
Copy the selected `model.joblib` into `03_deployment/saved_models/` before
building a standalone inference image.
