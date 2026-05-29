# 03 Deployment

This stage contains the deployment-focused version of the project.

Layout:

- `app/`: FastAPI application modules
- `src/data/`: config loading helpers used at inference time
- `src/features/`: feature engineering and preprocessing helpers
- `src/models/`: prediction logic for bundled or MLflow models
- `config/`: model and build configuration files
- `saved_models/`: optional bundled model artifacts for standalone serving

The deployment service can serve either a selected MLflow run or a bundled local
model.

If `RUN_ID` is set, the API loads `runs:/<run_id>/final_model` from MLflow. If
`RUN_ID` is not set, the API falls back to `saved_models/model.joblib`.

For Docker Compose, put the selected run ID in
`03_deployment/config/deployment.env`. Use `deployment.env.example` as the
template.

For local Docker usage, the root `docker-compose.yml` starts:

- an `mlflow` service backed by `./mlflow`
- an `api` service that serves the bundled model by default
- a `train` service that runs the training pipeline against the MLflow service

`02_model_training/models/compare_models.py` exports the active champion model.
Copy the selected `model.joblib` into `03_deployment/saved_models/` before
building a standalone inference image.
