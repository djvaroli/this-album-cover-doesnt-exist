"""
Implements different generator classes
"""


import typing

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Reshape


class UpSamplingBlock(tf.Module):
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
            activation: str = None
    ):
        super(UpSamplingBlock, self).__init__()
        self.n_channels = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation

        self.layers = [
            Conv2DTranspose(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                activation=activation
            ),
            BatchNormalization(),
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
            "use_bias": self.use_bias,
            "activation": self.activation
        }

    def __call__(self, inputs, *args, **kwargs):
        """
        Forward pass through the layer.
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs


class RGBImageGenerator(tf.Module):
    """
    Implements a vanilla GAN RGB image generator.
    """

    def __init__(
            self,
            embedding_dimension: int = 8 * 8 * 32,
            reshape_into: typing.Tuple = (8, 8, 32),
            name: str = "rgb_image_generator"
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param name:
        """
        super(RGBImageGenerator, self).__init__(name=name)
        self.embedding_dimension = embedding_dimension
        self.n_channels = 3  # rbg images
        self.reshape_into = reshape_into

        self.embedding_dense = Dense(embedding_dimension)
        self.convolutional_blocks = [
            UpSamplingBlock(512),  # 8, 8
            UpSamplingBlock(256, strides=(2, 2)),  # 16, 16
            UpSamplingBlock(128, strides=(2, 2)),  # 32, 32
            UpSamplingBlock(64, strides=(2, 2)),  # 64, 64
            UpSamplingBlock(32, strides=(2, 2)),  # 128, 128
            UpSamplingBlock(self.n_channels, strides=(2, 2), activation="sigmoid")  # 256, 256
        ]

    def __call__(self, inputs, *args, **kwargs):
        """
        Forward pass through the generator
        :param args:
        :param kwargs:
        :return:
        """

        outputs = inputs
        outputs = self.embedding_dense(outputs)  # embed the inputs
        outputs = Reshape(self.reshape_into)(outputs)
        for block in self.convolutional_blocks:
            outputs = block(outputs, *args, **kwargs)

        return outputs
        
