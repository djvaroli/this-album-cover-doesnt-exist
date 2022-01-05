"""
Implements different generator classes
"""


import typing
import yaml

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    Dense,
    Reshape,
    Add,
)

from ml.model_components.common import TFModelExtension


def _load_layer_config(filepath: str) -> dict:
    with open(filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _get_layers_from_config(configuration: dict):
    model_layers = []
    layer_configuration = configuration.copy()["layers"]
    for layer_config_ in layer_configuration:
        type_ = layer_config_.pop("type")
        layer = getattr(layers, type_)(**layer_config_)
        model_layers.append(layer)
    return model_layers


def _make_model_from_layers(input_layer: layers.Input, deep_layers: typing.List[layers.Layer]):
    o = input_layer
    for layer in deep_layers:
        o = layer(o)
    return Model(input_layer, o)


config = _load_layer_config(str(p))
model_layers = _get_layers_from_config(config)


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
