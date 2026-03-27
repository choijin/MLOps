import logging
from typing import Any, Callable

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def compare_and_promote_candidate(
    test_x: Any,
    test_y: Any,
    metric_fn: Callable[[Any, Any], float],
    tracking_uri: str,
    model_name: str,
    candidate_alias: str = "candidate",
    champion_alias: str = "champion",
    delete_candidate_on_promotion: bool = False,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    candidate_version = client.get_model_version_by_alias(
        model_name, candidate_alias
    ).version
    candidate_model = mlflow.sklearn.load_model(
        f"models:/{model_name}@{candidate_alias}"
    )
    candidate_score = float(metric_fn(test_y, candidate_model.predict(test_x)))

    champion_version = None
    champion_score = None

    try:
        champion_version = client.get_model_version_by_alias(
            model_name, champion_alias
        ).version
        champion_model = mlflow.sklearn.load_model(
            f"models:/{model_name}@{champion_alias}"
        )
        champion_score = float(metric_fn(test_y, champion_model.predict(test_x)))
    except Exception:
        logging.info(
            f"No current champion found for model '{model_name}'. Candidate v{candidate_version} will be promoted."
        )

    candidate_won = champion_score is None or candidate_score < champion_score

    champion_score_text = "N/A" if champion_score is None else f"{champion_score:.4f}"
    logging.info(
        f"Model comparison complete. Champion RMSE: {champion_score_text} | Candidate RMSE: {candidate_score:.4f}"
    )

    if candidate_won:
        client.set_registered_model_alias(model_name, champion_alias, candidate_version)
        logging.info(
            f"Promoted candidate v{candidate_version} to alias '{champion_alias}'."
        )
        if delete_candidate_on_promotion:
            client.delete_registered_model_alias(model_name, candidate_alias)
            logging.info(
                f"Removed alias '{candidate_alias}' after promotion.",
            )
    else:
        logging.info(
            f"Champion v{champion_version} remains active. Candidate v{candidate_version} was not promoted."
        )

    return {
        "candidate_version": int(candidate_version),
        "candidate_rmse": candidate_score,
        "champion_version": (int(champion_version)),
        "champion_rmse": champion_score,
        "candidate_won": candidate_won,
        "promoted_version": (
            int(candidate_version)
            if candidate_won
            else (int(champion_version) if champion_version is not None else None)
        ),
    }
