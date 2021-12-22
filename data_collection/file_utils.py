import typing
import json
from pathlib import Path


def save_json(filepath: str, data) -> None:
    """
    Saves data in a json file, under specified filepath.
    :param filepath:
    :param data:
    :return:
    """
    if filepath.endswith(".json") is False:
        filepath = f"{filepath}.json"

    with open(filepath, "w+") as f:
        json.dump(data, f)


def load_json(filepath: str):
    """
    Loads data from a json file.
    :param filepath:
    :return:
    """
    with open(filepath, "r+") as f:
        data = json.load(f)

    return data