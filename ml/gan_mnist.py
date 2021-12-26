"""
Train a GAN to generate the mnist dataset.
No text prompts just images.
"""

import tensorflow as tf
from tqdm import tqdm

from ml.model_components import generators, discriminators
from ml.training.losses import generator_loss_w_noise, discriminator_loss_w_noise
from ml.training.contexts import BaseModelTrainingContext



# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, context: BaseModelTrainingContext):
    noise = tf.random.normal([context.batch_size, context.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = context.generator(noise, training=True)

      real_output = context.discriminator(images, training=True)
      fake_output = context.discriminator(generated_images, training=True)

      gen_loss = context.generator_loss(fake_output)
      disc_loss = context.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, context.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, context.discriminator.trainable_variables)

    context.generator_optimizer.apply_gradients(zip(gradients_of_generator, context.generator.trainable_variables))
    context.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, context.discriminator.trainable_variables))
    

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
    
    
    


