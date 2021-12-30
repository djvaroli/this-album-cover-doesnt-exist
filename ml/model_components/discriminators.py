"""
Implements different discriminator classes
"""

import typing

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Dropout, Flatten

from ml.model_components.common import TFModelExtension


class DownSamplingBlock(TFModelExtension):
    """
    Applies convolutional down-sampling, followed by Dropout and LeakyRelu activation
    """

    def __init__(
            self,
            n_filters: int,
            kernel_size: typing.Tuple = (5, 5),
            strides: typing.Tuple = (1, 1),
            padding: str = "same",
            use_bias: bool = False,
            dropout_probability: float = 0.3
    ):
        super(DownSamplingBlock, self).__init__()
        self.n_channels = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        self.layers_ = [
            Conv2D(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias
            ),
            Dropout(dropout_probability),
            LeakyReLU()
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
            "use_bias": self.use_bias
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


class ImageDiscriminator(TFModelExtension):
    """
    Implements a basic image discriminator
    """

    def __init__(
            self,
            output_dense_activation: str = None,
            name: str = "rgb_image_discriminator"
    ):
        super(ImageDiscriminator, self).__init__(name=name)
        self.output_dense_activation = output_dense_activation

        self.down_sampling_blocks = [
            DownSamplingBlock(64, strides=(2, 2)),
            DownSamplingBlock(128, strides=(2, 2)),
            DownSamplingBlock(256, strides=(2, 2))
        ]
        self.output_dense = Dense(1, activation=output_dense_activation)

    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, list):
            outputs = inputs[0]
        else:
            outputs = inputs

        for block in self.down_sampling_blocks:
            outputs = block(outputs)

        outputs = Flatten()(outputs)
        return self.output_dense(outputs)

    def get_config(self) -> dict:
        """Returns class configuration dictionary to enable serialization

        Returns:

        """
        config: dict = super(ImageDiscriminator, self).get_config()
        config.update({
            "output_dense_activation": self.output_dense_activation
        })
        return config