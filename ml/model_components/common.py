import abc
from abc import ABC

import tensorflow as tf


def dict_to_kwarg_str(d: dict) -> str:
    """
    Given a dictionary returns a string of key-value pairs,
    i.e. given {"key": "value"} - returns "key=value".
    Assumes that d is a single-depth dictionary.
    :param d:
    :return:
    """
    s = ""
    for k, v in d.items():
        if isinstance(v, dict):
            raise Exception("Does not supported nested dictionaries.")
        s += f"{k}={v},"

    return s[:-1]


class TFModuleExtension(tf.Module, ABC):
    """
    Extends the tf.Module class with helper functionality.
    """

    def __repr__(self):
        class_name = self.__class__.__name__
        config_as_string = dict_to_kwarg_str(self.get_config())
        s = f"{class_name}({config_as_string})"
        return s

    @abc.abstractmethod
    def get_config(self) -> dict:
        """
        Returns the configuration for saving and loading.
        :return:
        """
        pass
