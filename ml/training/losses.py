import typing

import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def _add_normal_noise(
    inputs, sttdev: float = 0.01, clip_min: float = 0.0, clip_max: float = 1.0
):
    return tf.clip_by_value(
        tf.random.normal(shape=inputs.shape, stddev=sttdev) + inputs, clip_min, clip_max
    )


def discriminator_loss(real_output, fake_output) -> tf.Tensor:
    """
    Loss function for vanilla discriminator
    :param real_output:
    :param fake_output:
    :return:
    """

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = 1 / 2 * real_loss + 1 / 2 * fake_loss
    return total_loss


def generator_loss(fake_output):
    """
    Vanilla generator loss function
    :param fake_output:
    :return:
    """

    return cross_entropy(tf.ones_like(fake_output), fake_output)


def smoothed_discriminator_loss(real_output, fake_output):
    """
    A loss function for a discriminator that applies smoothing to positive labels
    in the form of random samples from a uniform distribution in the interval [0.05, 0.2].
    Slight modification of work from https://arxiv.org/pdf/1606.03498.pdf
    :param real_output:
    :param fake_output:
    :return:
    """
    smoothed_ones = tf.ones_like(real_output) - tf.random.uniform(
        real_output.shape, minval=0.05, maxval=0.2
    )
    real_loss = cross_entropy(smoothed_ones, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = 1 / 2 * real_loss + 1 / 2 * fake_loss
    return total_loss


def generator_loss_w_noise(fake_output):
    """
    Basic Generator loss but adds noise to labels.
    :param fake_output:
    :return:
    """
    return generator_loss(_add_normal_noise(fake_output))
