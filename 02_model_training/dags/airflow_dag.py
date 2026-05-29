from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/Users/jinchoi/Documents/Python/mlops"
PYTHON_BIN = f"{PROJECT_DIR}/.venv/bin/python"
TRAINING_DIR = f"{PROJECT_DIR}/02_model_training"
SPEC_PATH = "config/model_build_spec.json"
TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "zoomcamp-model"
REGISTERED_MODEL_NAME = "nyc-taxi-ridge"
CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"
TRAIN_START_YEAR = 2024


default_args = {
    "owner": "jinchoi",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="nyc_taxi_model_retraining",
    description="Train a candidate model, then compare it against the current champion",
    default_args=default_args,
    start_date=datetime(2024, 2, 1),
    schedule="@monthly",
    catchup=True,
    max_active_runs=1,
    tags=["mlops", "training", "mlflow"],
) as dag:
    # For a monthly schedule, data_interval_start points to the month we are training through.
    # Example: if the run fires on 2026-04-01, data_interval_start is 2026-03-01.
    run_training_pipeline = BashOperator(
        task_id="run_training_pipeline",
        cwd=TRAINING_DIR,
        bash_command=f"""
        {PYTHON_BIN} pipeline.py \
          --train-start-year {TRAIN_START_YEAR} \
          --train-year {{{{ data_interval_start.year }}}} \
          --train-end-month {{{{ data_interval_start.month }}}} \
          --spec-path {SPEC_PATH} \
          --tracking-uri {TRACKING_URI} \
          --experiment-name {EXPERIMENT_NAME} \
          --registered-model-name {REGISTERED_MODEL_NAME} \
          --candidate-alias {CANDIDATE_ALIAS} \
          --champion-alias {CHAMPION_ALIAS}
        """,
    )
