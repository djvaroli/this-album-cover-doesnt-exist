"""
    A list of utility functions for working with images
"""

import typing

import numpy as np
import tensorflow as tf
from PIL import Image


def arr_to_rgb_range(arr: np.ndarray) -> np.ndarray:
    """
    Given an array with values in an arbitrary range, scales pixel values to be in valid format
    for an image, i.e. each pixel value is rescaled to be in 0 -> 255 range.
    Args:
        arr:
        rgb_mode:

    Returns:

    """
    min_ = np.min(arr)
    max_ = np.max(arr)
    in_rgb_range = 255. * (arr - min_) / (max_ - min_)
    return in_rgb_range.astype("int16")


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

    return tf.keras.preprocessing.image.array_to_img(arr, *args, **kwargs)