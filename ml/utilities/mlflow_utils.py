import typing

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


def get_client_and_run_for_experiment(experiment_name: str) -> typing.Tuple[MlflowClient, Run]:
    """

    Args:
        experiment_name:

    Returns:

    """
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)

    run = client.create_run(experiment.experiment_id)

    return client, run
