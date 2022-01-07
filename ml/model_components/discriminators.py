"""
Implements different discriminator classes
"""

import typing

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Dropout, Flatten, Concatenate

from ml.model_components.common import TFModelExtension


class DownSamplingBlock(TFModelExtension):
    """
    Applies convolutional down-sampling, followed by Dropout and LeakyRelu activation
    """

    def __init__(
        self,
        n_filters: int,
        kernel_size: typing.Tuple = (4, 4),
        strides: typing.Tuple = (1, 1),
        padding: str = "same",
        use_bias: bool = False,
        dropout_probability: float = 0.3,
    ):
        super(DownSamplingBlock, self).__init__()
        self.n_channels = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.dropout_probability = dropout_probability

        self.layers_ = [
            Conv2D(
                n_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
            ),
            Dropout(dropout_probability),
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
            "dropout_probability": self.dropout_probability,
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
        name: str = "image_discriminator",
        add_input_noise: bool = False,
        kernel_size: tuple = (5, 5)
    ):
        super(ImageDiscriminator, self).__init__(name=name)
        self.kernel_size = kernel_size

        self.output_dense_activation = output_dense_activation
        self.add_input_noise = add_input_noise
        self.down_sampling_blocks = [
            DownSamplingBlock(64, strides=(2, 2), kernel_size=kernel_size),
            DownSamplingBlock(128, strides=(2, 2), kernel_size=kernel_size),
            DownSamplingBlock(256, strides=(2, 2), kernel_size=kernel_size),
        ]
        self.output_dense = Dense(1, activation=output_dense_activation)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        if isinstance(inputs, list):
            outputs = inputs[0]
        else:
            outputs = inputs

        if self.add_input_noise:
            outputs += tf.random.normal(outputs.shape)

        for block in self.down_sampling_blocks:
            outputs = block(outputs)

        outputs = Flatten()(outputs)
        return self.output_dense(outputs)

    def get_config(self) -> dict:
        """Returns class configuration dictionary to enable serialization

        Returns:

        """
        config: dict = super(ImageDiscriminator, self).get_config()
        config.update({"output_dense_activation": self.output_dense_activation})
        return config


class ConditionalImageDiscriminator(ImageDiscriminator):
    """Implements an image discriminator that takes as input a Tensor corresponding to
    either a generated or true image

    """

    def __init__(
        self,
        output_dense_activation: str = None,
        kernel_size: tuple = (4, 4),
        prompt_embedding_dim: int = 128,
        add_input_noise: bool = False,
        name: str = "conditional_image_discriminator",
    ):
        super(ImageDiscriminator, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.prompt_embedding_dim = prompt_embedding_dim
        self.output_dense_activation = output_dense_activation
        self.add_input_noise = add_input_noise
        self.down_sampling_blocks = [
            DownSamplingBlock(64, strides=(2, 2), kernel_size=kernel_size),
            DownSamplingBlock(128, strides=(2, 2), kernel_size=kernel_size),
            DownSamplingBlock(256, strides=(2, 2), kernel_size=kernel_size),
        ]

        self.prompt_dense = Dense(prompt_embedding_dim, activation="relu", use_bias=False)

        self.unit_convolution = DownSamplingBlock(256, strides=(1, 1), kernel_size=(1, 1))
        self.output_convolution = DownSamplingBlock(256, strides=(1, 1), kernel_size=kernel_size)
        self.output_dense = Dense(1, activation=output_dense_activation)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        input_image, prompt = inputs

        outputs = input_image
        if self.add_input_noise:
            outputs += tf.random.normal(outputs.shape)

        for block in self.down_sampling_blocks:
            outputs = block(outputs)

        # shape (batch_size, 1, 1, prompt_embedding_dim)
        prompt = self.prompt_dense(prompt)[:, tf.newaxis, tf.newaxis, :]
        repeated_prompt = tf.repeat(tf.repeat(prompt, repeats=4, axis=1), repeats=4, axis=2)

        outputs = Concatenate(axis=-1)([outputs, repeated_prompt])
        outputs = self.output_convolution(self.unit_convolution(outputs))
        outputs = Flatten()(outputs)
        return self.output_dense(outputs)