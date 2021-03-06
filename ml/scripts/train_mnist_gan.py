"""
Train a GAN to generate the mnist dataset.
No text prompts just images.
"""
from argparse import ArgumentParser
import pathlib
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = pathlib.Path(SCRIPT_DIR).parent.parent
sys.path.append(str(ROOT))

import tensorflow as tf
from tqdm import tqdm

from ml.model_components.generators import ImageGenerator
from ml.model_components.discriminators import ImageDiscriminator
from ml.training.losses import (
    smoothed_discriminator_loss,
    generator_loss,
    discriminator_loss,
)
from ml.training.contexts import (
    MNISTGANContext,
    GeneratorNamespace,
    DiscriminatorNamespace,
)
from ml.utilities import image_utils, mlflow_utils
from ml.training.data import get_mnist_dataset


EXPERIMENT_NAME = "GAN MNIST"
PROCESSING_OPS = {
    "normalize": lambda x: (x - 255.0) / 255.0,
    "unit_range": lambda x: (x - 127.5) / 127.5,
}


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

        real_images_predictions = discriminator_namespace.model(
            discriminator_inputs, training=True
        )
        generated_images_predictions = discriminator_namespace.model(
            generated_images, training=True
        )

        generator_step_loss = generator_namespace.loss_fn(generated_images_predictions)
        discriminator_step_loss = discriminator_namespace.loss_fn(
            real_images_predictions, generated_images_predictions
        )

    gradients_of_generator = generator_tape.gradient(
        generator_step_loss, generator_namespace.model.trainable_variables
    )
    gradients_of_discriminator = discriminator_tape.gradient(
        discriminator_step_loss, discriminator_namespace.model.trainable_variables
    )

    generator_namespace.optimizer.apply_gradients(
        zip(gradients_of_generator, generator_namespace.model.trainable_variables)
    )
    discriminator_namespace.optimizer.apply_gradients(
        zip(
            gradients_of_discriminator,
            discriminator_namespace.model.trainable_variables,
        )
    )

    step_loss = {
        "generator_loss": generator_step_loss,
        "discriminator_loss": discriminator_step_loss,
    }
    return step_loss


def get_train_context(
    batch_size: int,
    noise_dimension: int,
    epochs: int,
    label_smoothing: bool = False,
    pre_processing: str = "unit_range",
    discriminator_noise: bool = False,
) -> MNISTGANContext:
    """
    Prepares and returns the MNISTGANContext to be used in the training procedure.
    Returns:

    """

    generator_namespace = GeneratorNamespace(
        model=ImageGenerator(
            initial_filters=128,
            output_image_size=28,
            reshape_into=(7, 7, 256),
            embedding_dimension=7 * 7 * 256,
        ),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=generator_loss,
    )

    d_loss = discriminator_loss
    if label_smoothing:
        d_loss = smoothed_discriminator_loss

    discriminator_namespace = DiscriminatorNamespace(
        model=ImageDiscriminator(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=d_loss,
    )

    return MNISTGANContext(
        batch_size=batch_size,
        noise_dimension=noise_dimension,
        epochs=epochs,
        pre_processing=pre_processing,
        label_smoothing=label_smoothing,
        discriminator_noise=discriminator_noise,
        generator_namespace=generator_namespace,
        discriminator_namespace=discriminator_namespace,
    )


def train_gan(
    batch_size: int = 64,
    noise_dimension: int = 1024,
    epochs: int = 10,
    label_smoothing: bool = False,
    discriminator_noise: bool = False,
    pre_processing: str = "unit_range",
):
    """

    Args:
        batch_size:
        noise_dimension:
        epochs:
        label_smoothing: Whether to use smooth labels when calculating discriminator loss
        discriminator_noise: Whether to add noise to discriminator inputs
        pre_processing: The kind of processing to apply to the MNIST images


    Returns:

    """
    processing_op = PROCESSING_OPS.get(pre_processing)
    mlflow_client, run = mlflow_utils.get_client_and_run_for_experiment(EXPERIMENT_NAME)

    mlflow_client.log_params(
        run.info.run_id,
        batch_size=batch_size,
        noise_dimension=noise_dimension,
        epochs=epochs,
        label_smoothing=label_smoothing,
        pre_processing=pre_processing,
        discriminator_noise=discriminator_noise,
    )

    data = get_mnist_dataset(batch_size=batch_size, preprocess_fn=processing_op)
    context = get_train_context(
        batch_size=batch_size,
        noise_dimension=noise_dimension,
        epochs=epochs,
        label_smoothing=label_smoothing,
        discriminator_noise=discriminator_noise,
        pre_processing=pre_processing,
    )

    # this will have 10 samples, instead of the number of batches
    reference = context.set_reference()
    reference_images = image_utils.make_image_grid(
        context.generator_namespace.model(reference, training=False)
    )
    reference_images = image_utils.array_to_image(reference_images)
    mlflow_client.log_image(run.info.run_id, reference_images, f"epoch_0.png")

    for epoch in range(1, epochs + 1):
        for image_batch in tqdm(data):
            generator_input_noise = context.generate_noise()
            context.assign_inputs(generator_input_noise, image_batch)
            step_loss = train_step(context)

            mlflow_client.log_metric(
                run.info.run_id,
                "discriminator_loss",
                step_loss["generator_loss"].numpy(),
            )
            mlflow_client.log_metric(
                run.info.run_id,
                "generator_loss",
                step_loss["discriminator_loss"].numpy(),
            )

        model_prediction = context.generator_namespace.model(reference, training=False)

        model_prediction = image_utils.tensor_to_rgb_range(model_prediction)
        reference_images = image_utils.make_image_grid(model_prediction)
        reference_images = image_utils.array_to_image(reference_images)
        mlflow_client.log_image(run.info.run_id, reference_images, f"epoch_{epoch}.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=64, help="The size of each batch of images."
    )
    parser.add_argument(
        "--noise_dimension",
        type=int,
        default=1024,
        help="The size of the noise fed to the generator.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train the model for."
    )
    parser.add_argument(
        "--label_smoothing",
        default=False,
        action="store_true",
        help="Add smoothing to labels fed to discriminator.",
    )
    parser.add_argument(
        "--discriminator_noise",
        default=False,
        action="store_true",
        help="Add noise to inputs to the discriminator.",
    )
    parser.add_argument(
        "--pre_processing",
        default="unit_range",
        help="Kind of pre-processing to apply to image [normalize, unit_range]",
        type=str,
    )

    args = parser.parse_args().__dict__
    train_gan(**args)
