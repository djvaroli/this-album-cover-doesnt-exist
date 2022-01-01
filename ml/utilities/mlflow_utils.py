import typing

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


class ExtendedMLflowClient(MlflowClient):
    """
    Extends certain functionality of MLflowClient class
    """

    def __init__(self, *args, **kwargs):
        super(ExtendedMLflowClient, self).__init__(*args, **kwargs)

    def log_params(self, run_id: str, **kwargs):
        """Logs multiple params in one call

        Args:
            run_id:
            **kwargs:

        Returns:

        """
        for k, v in kwargs.items():
            super(ExtendedMLflowClient, self).log_param(run_id, k, v)


def get_client_and_run_for_experiment(experiment_name: str) -> typing.Tuple[ExtendedMLflowClient, Run]:
    """

    Args:
        experiment_name:

    Returns:

    """
    client = ExtendedMLflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)

    run = client.create_run(experiment.experiment_id)

    return client, run
