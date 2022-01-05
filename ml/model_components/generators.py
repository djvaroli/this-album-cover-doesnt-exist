"""
Implements different generator classes
"""


import typing

import numpy as np
from tensorflow.keras.layers import (
    Dense,
    Reshape,
    Add,
)

from ml.model_components.layers import UpSamplingBlock
from ml.model_components.common import TFModelExtension


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
            embedding_dimension, activation=initial_dense_activation
        )

        # this assumes a few things about shapes  TODO add tests to make ensure compatible shapes
        n_upsmpl_blocks = int(np.log2(output_image_size // reshape_into[0]))
        self.up_sampling_blocks = []
        filters = initial_filters
        for _ in range(n_upsmpl_blocks):
            self.up_sampling_blocks.append(UpSamplingBlock(filters, strides=(2, 2)))
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


class RGBImageGenerator(ImageGenerator):
    """
    Vanilla generator to be used in a GAN
    """

    def __init__(
        self,
        embedding_dimension: int = 8 * 8 * 32,
        reshape_into: typing.Tuple = (8, 8, 32),
        output_image_size: int = 256,
        initial_filters: int = 512,
        name: str = "rgb_image_generator",
        output_activation: str = "sigmoid",
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param output_image_size: The dimensions of the generated images (assumes images are square)
        :param initial_filters: the number of filters in the first UpSampling block
        :param name:
        :param output_activation:
        """
        super(RGBImageGenerator, self).__init__(
            embedding_dimension=embedding_dimension,
            reshape_into=reshape_into,
            output_image_size=output_image_size,
            initial_filters=initial_filters,
            name=name,
            n_channels=3,
            output_activation=output_activation,
        )


class RGBImageGeneratorWithTextPrompt(RGBImageGenerator):
    """
    Implements a vanilla GAN RGB image generator with the addition of accepting two separate tensors as inputs.
    The first tensor is the "noise" and the second tensor is an embedding of a text prompt.
    The model will first pass both through a separate dense layer and then combine by addition.
    """

    def __init__(
        self,
        embedding_dimension: int = 8 * 8 * 32,
        reshape_into: typing.Tuple = (8, 8, 32),
        output_image_size: int = 256,
        initial_filters: int = 512,
        name: str = "rgb_image_generator_w_text_prompt",
        output_activation: str = "sigmoid",
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param name:
        :param output_activation:
        """
        super(RGBImageGeneratorWithTextPrompt, self).__init__(
            name=name,
            embedding_dimension=embedding_dimension,
            reshape_into=reshape_into,
            output_image_size=output_image_size,
            initial_filters=initial_filters,
            output_activation=output_activation,
        )

        self.noise_dense = Dense(embedding_dimension, activation="relu")
        self.text_prompt_dense = Dense(embedding_dimension, activation="relu")

    def call(self, inputs, *args, **kwargs):
        """
        Forward pass through the generator
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        noise_input, text_input = inputs
        noise_input = self.noise_dense(noise_input)
        text_input = self.text_prompt_dense(text_input)
        outputs = Add([noise_input, text_input])

        return super(RGBImageGeneratorWithTextPrompt, self).call(
            outputs, *args, **kwargs
        )
