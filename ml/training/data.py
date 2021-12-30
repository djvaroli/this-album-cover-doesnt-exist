"""
Functions related to obtaining and pre-processing / formatting data for training models.
"""

from typing import Callable

import tensorflow as tf


def get_mnist_dataset(
        buffer_size: int = 60000,
        batch_size: int = 256,
        preprocess_fn: Callable = None
) -> tf.data.Dataset:
    """Returns an instance of tf.data.Dataset containing the MNIST images.

    Args:
        buffer_size:
        batch_size:
        preprocess_fn: Function to apply to dataset

    Returns:

    """
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    if preprocess_fn is not None:
        train_images = preprocess_fn(train_images)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
    return train_dataset


