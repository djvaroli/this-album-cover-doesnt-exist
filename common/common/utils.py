import typing
import json


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


def batch_generator(list_: typing.List, batch_size: int = 10) -> typing.Generator:
    """
    Yield successive batches from a specified list
    :param list_:
    :param batch_size:
    :return:
    """
    for i in range(0, len(list_), batch_size):
        yield list_[i:i + batch_size]


def list_into_batches(list_: typing.List, batch_size: int = 10) -> typing.List[typing.List]:
    """
    Given a list of elements, converts into a nested list with successive batches.
    :param list_:
    :param batch_size:
    :return:
    """
    return list(batch_generator(list_, batch_size))