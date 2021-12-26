import typing

import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def _add_normal_noise(inputs, sttdev: float = 0.01, clip_min: float = 0.0, clip_max: float = 1.0):
    return tf.clip_by_value(tf.random.normal(shape=inputs.shape, stddev=sttdev) + inputs, clip_min, clip_max)


def discriminator_loss(real_output, fake_output, weights: typing.List[float] = None):
    """
    Loss function for vanilla discriminator
    :param real_output:
    :param fake_output:
    :return:
    """

    if weights is None:
        weights = [0.5, 0.5]

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = weights[0] * real_loss + weights[1] * fake_loss
    return total_loss


def generator_loss(fake_output):
    """
    Vanilla generator loss function
    :param fake_output:
    :return:
    """

    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss_w_noise(real_output, fake_output):
    """
    Discriminator loss that applies normally distributed noise to labels
    :param real_output:
    :param fake_output:
    :return:
    """
    real_output = _add_normal_noise(real_output)
    fake_output = _add_normal_noise(fake_output)
    return discriminator_loss(real_output, fake_output)


def generator_loss_w_noise(fake_output):
    """
    Basic Generator loss but adds noise to labels.
    :param fake_output:
    :return:
    """
    return generator_loss(_add_normal_noise(fake_output))