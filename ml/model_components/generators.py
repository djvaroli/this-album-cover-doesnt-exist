"""
Implements different generator classes
"""


import typing

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Reshape, Add

from ml.model_components import TFModuleExtension


class UpSamplingBlock(TFModuleExtension):
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


class ImageGenerator(tf.Module):
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
            output_activation: str = "sigmoid"
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param output_image_size: The dimensions of the generated images (assumes images are square)
        :param initial_filters: the number of filters in the first UpSampling block
        :param n_channels:
        :param name:
        :param output_activation: 
        """
        super(ImageGenerator, self).__init__(name=name)
        self.n_channels = n_channels
        self.embedding_dimension = embedding_dimension
        self.output_image_size = output_image_size
        self.output_image_shape = (1, self.n_channels, self.output_image_size, self.output_image_size)
        self.reshape_into = reshape_into

        self.initial_dense = Dense(embedding_dimension, activation="relu")

        # this assumes a few things about shapes  TODO add tests to make ensure compatible shapes
        n_upsmpl_blocks = int(np.log2(output_image_size // reshape_into[0]))
        self.up_sampling_blocks = []
        filters = initial_filters
        for _ in range(n_upsmpl_blocks):
            self.up_sampling_blocks.append(UpSamplingBlock(filters, strides=(2, 2)))
            filters //= 2

        self.up_sampling_blocks.append(UpSamplingBlock(n_filters=self.n_channels, strides=(1, 1), activation=output_activation))

    def __call__(self, inputs, *args, **kwargs):
        """
        Forward pass through the generator
        :param args:
        :param kwargs:
        :return:
        """
        outputs = self.initial_dense(inputs)
        outputs = Reshape(self.reshape_into)(outputs)
        for block in self.up_sampling_blocks:
            outputs = block(outputs, *args, **kwargs)

        return outputs


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
            output_activation: str = "sigmoid"
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
            output_activation=output_activation
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
            output_activation: str = "sigmoid"
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
            output_activation=output_activation
        )

        self.noise_dense = Dense(embedding_dimension, activation="relu")
        self.text_prompt_dense = Dense(embedding_dimension, activation="relu")

    def __call__(self, inputs, *args, **kwargs):
        """
        Forward pass through the generator
        :param args:
        :param kwargs:
        :return:
        """
        noise_input, text_input = inputs
        noise_input = self.noise_dense(noise_input)
        text_input = self.text_prompt_dense(text_input)
        outputs = Add([noise_input, text_input])

        return super(RGBImageGeneratorWithTextPrompt, self).__call__(outputs, *args, **kwargs)
        
