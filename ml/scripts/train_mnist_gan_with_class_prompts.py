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
import wandb

from ml.model_components.generators import RGBImageGeneratorWithTextPrompt
from ml.model_components.discriminators import ImageDiscriminator
from ml.training.losses import (
    smoothed_discriminator_loss,
    generator_loss,
    discriminator_loss,
)
from ml.training.contexts import (
    MnistPromptGANContext,
    GeneratorNamespace,
    DiscriminatorNamespace,
)
from ml.utilities import image_utils
from ml.training.data import get_mnist_dataset_with_labels


EXPERIMENT_NAME = "GAN MNIST"
PROCESSING_OPS = {
    "normalize": lambda x: (x - 255.0) / 255.0,
    "unit_range": lambda x: (x - 127.5) / 127.5,
}


def train_step(
        context: MnistPromptGANContext
) -> dict:
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


def train_gan(context: MnistPromptGANContext):
    """ """

    processing_op = PROCESSING_OPS.get(context.pre_processing)

    if processing_op is None:
        raise Exception("Specified invalid image pre-processing operation.")

    data = get_mnist_dataset_with_labels(batch_size=context.batch_size, preprocess_fn=processing_op)

    # this will have 10 samples, instead of the number of batches
    reference = context.set_reference()
    reference_images = image_utils.make_image_grid(
        context.generator_namespace.model(reference, training=False)
    )
    reference_images = image_utils.array_to_image(reference_images)

    wandb.log({"reference_image": wandb.Image(reference_images)}, step=0)

    for epoch in range(1, context.epochs + 1):
        step_loss = {}
        for image_batch, image_labels in tqdm(data):
            generator_input_noise = context.generate_noise()
            context.assign_inputs([generator_input_noise, image_labels], image_batch)
            step_loss = train_step(context)

        model_prediction = context.generator_namespace.model(reference, training=False)
        model_prediction = image_utils.tensor_to_rgb_range(model_prediction)
        reference_images = image_utils.make_image_grid(model_prediction)
        reference_images = image_utils.array_to_image(reference_images)

        wandb.log(
            {"discriminator_loss": step_loss["discriminator_loss"].numpy()}, step=epoch
        )
        wandb.log({"generator_loss": step_loss["generator_loss"].numpy()}, step=epoch)
        wandb.log({"reference_image": wandb.Image(reference_images)}, step=epoch)


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

    args = parser.parse_args()
    wandb.init(
        project="this-album-cover-doesnt-exist",
        entity="djvaroli",
        tags=["baseline", "gan", "mnist"],
    )
    wandb.config.update(args)

    # set up the generator
    generator_namespace = GeneratorNamespace(
        model=RGBImageGeneratorWithTextPrompt(
            initial_filters=128,
            output_image_size=28,
            reshape_into=(7, 7, 256),
            embedding_dimension=7 * 7 * 256,
        ),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=generator_loss,
    )

    # set up the discriminator
    d_loss = discriminator_loss
    if wandb.config.label_smoothing:
        d_loss = smoothed_discriminator_loss

    discriminator_namespace = DiscriminatorNamespace(
        model=ImageDiscriminator(add_input_noise=wandb.config.discriminator_noise),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=d_loss,
    )

    context = MnistPromptGANContext(
        batch_size=wandb.config.batch_size,
        noise_dimension=wandb.config.noise_dimension,
        epochs=wandb.config.epochs,
        label_smoothing=wandb.config.label_smoothing,
        pre_processing=wandb.config.pre_processing,
        discriminator_noise=wandb.config.discriminator_noise,
        generator_namespace=generator_namespace,
        discriminator_namespace=discriminator_namespace,
    )

    train_gan(context)
    wandb.finish()
