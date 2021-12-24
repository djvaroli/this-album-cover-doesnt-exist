"""
Implements different generator classes
"""


import typing

import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, LeakyReLU, Dense, Reshape, Add


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
    Vanilla generator to be used in a GAN
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
        self.n_channels = 3  # rgb images
        self.reshape_into = reshape_into

        self.initial_dense = Dense(embedding_dimension, activation="relu")

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
        outputs = self.initial_dense(inputs)
        outputs = Reshape(self.reshape_into)(outputs)
        for block in self.convolutional_blocks:
            outputs = block(outputs, *args, **kwargs)

        return outputs


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
            name: str = "rgb_image_generator_w_text_prompt"
    ):
        """
        Assumes that the generated images are squares.
        :param embedding_dimension:
        :param reshape_into:
        :param name:
        """
        super(RGBImageGeneratorWithTextPrompt, self).__init__(
            name=name, embedding_dimension=embedding_dimension, reshape_into=reshape_into
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
        
