"""
Implements different generator classes
"""


import typing

import numpy as np
from tensorflow.keras.layers import (
    Conv2DTranspose,
    BatchNormalization,
    ReLU,
    Dense,
    Reshape,
    Concatenate
)

from ml.model_components.common import TFModelExtension


class UpSamplingBlock(TFModelExtension):
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
            ReLU(),
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


class ImageGenerator(TFModelExtension):
    """
    Vanilla generator to be used in a GAN
    """

    def __init__(
        self,
        embedding_dimension: int = 8 * 8 * 32,
        reshape_into: typing.Tuple = (8, 8, 32),
        output_image_size: int = 256,
        initial_filters: int = 512,
        n_channels: int = 1,
        kernel_size: tuple = (4, 4),
        name: str = "image_generator",
        initial_dense_activation: str = None,
        output_activation: str = "tanh",
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param output_image_size: The dimensions of the generated images (assumes images are square)
        :param initial_filters: the number of filters in the first UpSampling block
        :param n_channels:
        :param name:
        :param initial_dense_activation:
        :param output_activation:
        :param kernel_size:
        """

        super(ImageGenerator, self).__init__(name=name)
        self.n_channels = n_channels
        self.initial_dense_activation = initial_dense_activation
        self.embedding_dimension = embedding_dimension
        self.output_image_size = output_image_size
        self.output_image_shape = (
            1,
            self.n_channels,
            self.output_image_size,
            self.output_image_size,
        )
        self.reshape_into = reshape_into
        self.output_activation = output_activation
        self.initial_filters = initial_filters

        self.initial_dense = Dense(
            embedding_dimension, activation=initial_dense_activation, use_bias=False
        )

        # this assumes a few things about shapes  TODO add tests to make ensure compatible shapes
        n_upsmpl_blocks = int(np.log2(output_image_size // reshape_into[0]))
        self.up_sampling_blocks = []
        filters = initial_filters
        for _ in range(n_upsmpl_blocks):
            self.up_sampling_blocks.append(
                UpSamplingBlock(filters, strides=(2, 2), kernel_size=kernel_size,
                )
            )
            filters //= 2

        self.up_sampling_blocks.append(
            UpSamplingBlock(n_filters=self.n_channels, activation=output_activation)
        )

    def call(self, inputs, *args, **kwargs):
        """
        Forward pass through the generator
        :param args:
        :param kwargs:
        :return:
        """
        if isinstance(inputs, list):
            outputs = inputs[0]
        else:
            outputs = inputs

        outputs = self.initial_dense(outputs)
        outputs = ReLU()(BatchNormalization()(outputs))
        outputs = Reshape(self.reshape_into)(outputs)
        for block in self.up_sampling_blocks:
            outputs = block(outputs, *args, **kwargs)

        return outputs

    def get_config(self) -> dict:
        """Returns a dictionary of configuration parameters

        Returns:

        """
        config: dict = super(ImageGenerator, self).get_config()
        config.update(
            {
                "name": self.name,
                "n_channels": self.n_channels,
                "initial_dense_activation": self.initial_dense_activation,
                "embedding_dimension": self.embedding_dimension,
                "output_image_size": self.output_image_size,
                "output_image_shape": self.output_image_shape,
                "reshape_into": self.reshape_into,
                "output_activation": self.output_activation,
                "initial_filters": self.initial_filters,
            }
        )
        return config


class ConditionalImageGenerator(ImageGenerator):
    """Implements an image generator class that is able to take as inputs a tensor of noise
    and a second tensor representing some condition to be used when generating the output image.
    Follows the procedure described in https://arxiv.org/pdf/1605.05396.pdf
    """

    def __init__(
        self,
        embedding_dimension: int = 8 * 8 * 32,
        reshape_into: typing.Tuple = (8, 8, 32),
        output_image_size: int = 256,
        initial_filters: int = 512,
        n_channels: int = 1,
        kernel_size: tuple = (4, 4),
        prompt_embedding_dim: int = 128,
        name: str = "conditional_image_generator",
        output_activation: str = "tanh",
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param name:
        :param output_activation:
        :param kernel_size:
        :param prompt_embedding_dim:
        """
        super(ConditionalImageGenerator, self).__init__(
            name=name,
            embedding_dimension=embedding_dimension,
            reshape_into=reshape_into,
            output_image_size=output_image_size,
            initial_filters=initial_filters,
            output_activation=output_activation,
            n_channels=n_channels,
            kernel_size=kernel_size
        )
        self.prompt_embedding_dim = prompt_embedding_dim
        self.prompt_embedding = Dense(prompt_embedding_dim, activation="relu")

    def call(self, inputs, *args, **kwargs):
        """
        Forward pass through the generator
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        noise_input, prompt = inputs
        prompt = self.prompt_embedding(prompt)
        concatenated = Concatenate()([noise_input, prompt])

        return super(ConditionalImageGenerator, self).call(
            concatenated, *args, **kwargs
        )