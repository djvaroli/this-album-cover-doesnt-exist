"""
Train a GAN to generate the mnist dataset.
No text prompts just images.
"""

import tensorflow as tf

from ml.model_components import generators, discriminators
from ml.training.losses import generator_loss_w_noise, discriminator_loss_w_noise


def train_gan():
    """

    :return:
    """

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_loss = generator_loss_w_noise
    discriminator_loss = discriminator_loss_w_noise

    generator = generators.ImageGenerator(initial_filters=128, output_image_size=28, reshape_into=(7, 7, 256))
    discriminator = discriminators.ImageDiscriminator()


