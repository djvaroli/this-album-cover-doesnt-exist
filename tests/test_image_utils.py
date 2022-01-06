
import tensorflow as tf


def test_make_image_grid():
    n_slices = 10
    image_size = 28
    n_channels = 1
    scaling_factors = tf.reshape(
        tf.cast(tf.linspace(0, 1, n_slices), "float32"), (n_slices, 1, 1, n_channels)
    )
    test_images = tf.tile(scaling_factors, (1, image_size, image_size, n_channels))

    assert test_images.shape == (
        n_slices,
        image_size,
        image_size,
        n_channels,
    ), "Image dimensions mismatch."

    unstacked_images = tf.unstack(test_images)
    grid_images = tf.concat(unstacked_images, axis=1)

    assert grid_images.shape == (image_size, image_size * n_slices, n_channels)