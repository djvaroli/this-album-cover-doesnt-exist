import typing
import yaml

from tensorflow.keras import layers as tf_layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, BatchNormalization, LeakyReLU


def _load_layer_config(filepath: str) -> dict:
    with open(filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _get_layers_from_config(configuration: dict):
    model_layers = []
    layer_configuration = configuration.copy()["layers"]
    for layer_config_ in layer_configuration:
        type_ = layer_config_.pop("type")
        layer = getattr(tf_layers, type_)(**layer_config_)
        model_layers.append(layer)
    return model_layers


class UpSamplingBlock(Layer):
    """
    Performs convolutional up-sampling, followed by batch-normalization and then a Leaky Relu activation function.
    """

    def __init__(
        self,
        n_filters: int,
        kernel_size: typing.Tuple = (5, 5),
        strides: typing.Tuple = (1, 1),
        padding: str = "same",
        use_bias: bool = False,
        activation: str = None,
    ):
        super(UpSamplingBlock, self).__init__()
        self.n_channels = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation

        self.layers_ = [
            Conv2DTranspose(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                activation=activation,
            ),
            BatchNormalization(),
            LeakyReLU(),
        ]

    def get_config(self):
        """
        Returns the configuration of the current layer
        :return:
        """

        return {
            "n_channels": self.n_channels,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "activation": self.activation,
        }

    def call(self, inputs, *args, **kwargs):
        """
        Forward pass through the layer.
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        outputs = inputs
        for layer in self.layers_:
            outputs = layer(outputs)

        return outputs