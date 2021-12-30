"""
    A list of utility functions for working with images
"""

import typing

import numpy as np
import tensorflow as tf
from PIL import Image


def make_image_grid(tensor: tf.Tensor) -> tf.Tensor:
    """Given a 4D tensor with dimensions [batch size, image height, image width, number of channels],
    returns a new tensor or array where each image has been stacked along the width axis.

    Args:
        tensor:

    Returns:

    """

    # TODO - add option to specify number of rows instead of fixing to 1

    image_slices = tf.unstack(tensor)
    image_grid = tf.concat(image_slices, axis=1)
    return image_grid


def array_to_image(arr: typing.Union[np.ndarray, tf.Tensor], *args, **kwargs) -> Image:
    """Wrapper around the tf.keras.utils.array_to_img method

    Args:
        arr: Array or Tensor to be converted to an image.

    Returns:

    """

    return tf.keras.utils.array_to_img(arr, *args, **kwargs)