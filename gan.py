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

    generator = generators.RGBImageGeneratorWithTextPrompt()
    discriminator = discriminators.RGBImageDiscriminator()
