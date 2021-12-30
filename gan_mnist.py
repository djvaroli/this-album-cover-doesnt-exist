"""
Train a GAN to generate the mnist dataset.
No text prompts just images.
"""
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from ml.model_components import generators, discriminators
from ml.training.losses import generator_loss_w_noise, discriminator_loss_w_noise, generator_loss, discriminator_loss
from ml.training.contexts import MNISTGANContext, GeneratorNamespace, DiscriminatorNamespace
from ml.utilities import image_utils, mlflow_utils
from ml.training.data import get_mnist_dataset


EXPERIMENT_NAME = "GAN MNIST"


def train_step(context: MNISTGANContext) -> dict:
    """

    Args:
        context:

    Returns:
        step_loss: dict - returns a dictionary containing the generator and discriminator loss after the step.
    """

    generator_namespace = context.generator_namespace
    discriminator_namespace = context.discriminator_namespace
    generator_inputs = generator_namespace.model_inputs
    discriminator_inputs = discriminator_namespace.model_inputs

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generated_images = generator_namespace.model(generator_inputs, training=True)
        
        real_images_predictions = discriminator_namespace.model(discriminator_inputs, training=True)
        generated_images_predictions = discriminator_namespace.model(generated_images, training=True)

        generator_step_loss = generator_namespace.loss_fn(generated_images_predictions)
        discriminator_step_loss = discriminator_namespace.loss_fn(real_images_predictions, generated_images_predictions)

    gradients_of_generator = generator_tape.gradient(generator_step_loss, generator_namespace.model.trainable_variables)
    gradients_of_discriminator = discriminator_tape.gradient(
        discriminator_step_loss, discriminator_namespace.model.trainable_variables
    )

    generator_namespace.optimizer.apply_gradients(
        zip(gradients_of_generator, generator_namespace.model.trainable_variables)
    )
    discriminator_namespace.optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator_namespace.model.trainable_variables)
    )

    step_loss = {
        "generator_loss": generator_step_loss,
        "discriminator_loss": discriminator_step_loss
    }
    return step_loss


def get_train_context(
        batch_size: int,
        noise_dimension: int,
        epochs: int,
        noisy_loss: bool = False
) -> MNISTGANContext:
    """
    Prepares and returns the MNISTGANContext to be used in the training procedure.
    Returns:

    """
    g_loss = generator_loss
    d_loss = discriminator_loss

    if noisy_loss:
        g_loss = generator_loss_w_noise
        d_loss = discriminator_loss_w_noise

    generator_namespace = GeneratorNamespace(
        model=generators.ImageGenerator(
            initial_filters=128,
            output_image_size=28,
            reshape_into=(7, 7, 256),
            embedding_dimension=7*7*256
        ),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=g_loss
    )

    discriminator_namespace = DiscriminatorNamespace(
        model=discriminators.ImageDiscriminator(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=d_loss
    )

    return MNISTGANContext(
        batch_size=batch_size,
        noise_dimension=noise_dimension,
        epochs=epochs,
        generator_namespace=generator_namespace,
        discriminator_namespace=discriminator_namespace
    )


def train_gan(
        batch_size: int = 64,
        noise_dimension: int = 1024,
        epochs: int = 10,
        noisy_loss: bool = False
):
    """

    Args:
        batch_size:
        noise_dimension:
        epochs:

    Returns:

    """

    mlflow_client, run = mlflow_utils.get_client_and_run_for_experiment(EXPERIMENT_NAME)

    mlflow_client.log_param(run.info.run_id, "batch_size", batch_size)
    mlflow_client.log_param(run.info.run_id, "noise_dimension", noise_dimension)
    mlflow_client.log_param(run.info.run_id, "epochs", epochs)
    mlflow_client.log_param(run.info.run_id, "noisy_loss", noisy_loss)

    data = get_mnist_dataset(batch_size=batch_size, preprocess_fn=lambda x: (x - 255.) / 255.)
    context = get_train_context(batch_size, noise_dimension, epochs, noisy_loss)

    # this will have 10 samples, instead of the number of batches
    reference = context.set_reference()
    reference_images = image_utils.make_image_grid(context.generator_namespace.model(reference, training=False))
    reference_images = image_utils.array_to_image(reference_images)
    mlflow_client.log_image(run.info.run_id, reference_images, f"epoch_0.png")

    for epoch in range(1, epochs + 1):
        for image_batch in tqdm(data):
            generator_input_noise = context.generate_noise()
            context.assign_inputs(generator_input_noise, image_batch)
            step_loss = train_step(context)

            mlflow_client.log_metric(run.info.run_id, "discriminator_loss", step_loss["generator_loss"].numpy())
            mlflow_client.log_metric(run.info.run_id, "generator_loss", step_loss["discriminator_loss"].numpy())

        reference_images = image_utils.make_image_grid(context.generator_namespace.model(reference, training=False))
        reference_images = image_utils.array_to_image(reference_images)
        mlflow_client.log_image(run.info.run_id, reference_images, f"epoch_{epoch}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of each batch of images."
    )
    parser.add_argument(
        "--noise_dimension", type=int, default=1024, help="The size of the noise fed to the generator."
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train the model for."
    )
    parser.add_argument(
        "--noisy_loss", default=False, action="store_true", help="Add noise in loss functions."
    )

    args = parser.parse_args().__dict__
    train_gan(**args)





    
    


