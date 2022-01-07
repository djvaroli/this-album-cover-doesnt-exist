import abc
import typing
from abc import ABC
import time

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


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


class _TFReprClass(ABC):
    """
    Standardizes the __repr__ method of a class
    """

    @abc.abstractmethod
    def get_config(self) -> dict:
        raise NotImplementedError()

    def __repr__(self):
        class_name = self.__class__.__name__
        config_as_string = dict_to_kwarg_str(self.get_config())
        s = f"{class_name}({config_as_string})"
        return s


class TFModelExtension(Model, _TFReprClass):
    """
    Extends base Tensorflow Model functionality
    """

    def __init__(self, input_shape: typing.Iterable = None, **kwargs):
        super(TFModelExtension, self).__init__(**kwargs)
        self.__model = None
        self.input_layer_shape = input_shape

    @abc.abstractmethod
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError

    @property
    def model(self):
        """Returns an instance of tf.keras.models.Model corresponding to the model graph defined by the class

        Returns:

        """
        if self.__model is None and self.input_layer_shape is not None:
            inputs = []
            input_layers_shapes = self.input_layer_shape

            # in the event that the user specified only one input layer
            if not isinstance(input_layers_shapes, list):
                input_layers_shapes = [input_layers_shapes]

            for input_shape_ in input_layers_shapes:
                inputs.append(Input(input_shape_))

            self.__model = Model(inputs, self.call(inputs, training=False))

        return self.__model

    def plot(
        self, to_file: str = None, show_shapes: bool = True, *args, **kwargs
    ) -> str:
        """Plots the model graph contained in the class

        Args:
            to_file:
            show_shapes:
            *args:
            **kwargs:

        Returns:

        """
        if to_file is None:
            to_file = f"{self.model.name}-{int(time.time())}.png"

        return tf.keras.utils.plot_model(
            self.model, to_file, show_shapes=show_shapes, *args, **kwargs
        )
