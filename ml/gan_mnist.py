"""
Train a GAN to generate the mnist dataset.
No text prompts just images.
"""

import tensorflow as tf
from tqdm import tqdm

from ml.model_components import generators, discriminators
from ml.training.losses import generator_loss_w_noise, discriminator_loss_w_noise
from ml.training.contexts import MNISTGANContext, GeneratorNamespace, DiscriminatorNamespace


def train_step(context: MNISTGANContext):
    """

    Args:
        context:

    Returns:

    """
    generator_namespace = context.generator_namespace
    discriminator_namespace = context.discriminator_namespace
    generator_inputs = context.model_inputs
    real_images = discriminator_namespace.image_batch

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generated_images = generator_namespace.model(generator_inputs, training=True)
        
        real_output = discriminator_namespace.model(real_images, training=True)
        fake_output = discriminator_namespace.model(generated_images, training=True)

        generator_loss = generator_namespace.loss_fn(fake_output)
        discriminator_loss = discriminator_namespace.loss_fn(real_output, fake_output)

    gradients_of_generator = generator_tape.gradient(generator_loss, generator_namespace.model.trainable_variables)
    gradients_of_discriminator = discriminator_tape.gradient(
        discriminator_loss, discriminator_namespace.model.trainable_variables
    )

    generator_namespace.optimizer.apply_gradients(
        zip(gradients_of_generator, generator_namespace.model.trainable_variables)
    )
    discriminator_namespace.optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator_namespace.model.trainable_variables)
    )


def train_gan():
    """

    :return:
    """
    generator_namespace = GeneratorNamespace(
        model=generators.ImageGenerator(initial_filters=128, output_image_size=28, reshape_into=(7, 7, 256)),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=generator_loss_w_noise
    )

    discriminator_namespace = DiscriminatorNamespace(
        model=discriminators.ImageDiscriminator(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=discriminator_loss_w_noise
    )

    context = MNISTGANContext(generator_namespace=generator_namespace, discriminator_namespace=discriminator_namespace)




    
    


