import logging

import tensorflow as tf
import numpy as np

from ml.model_components import generators, discriminators
from ml.training import contexts, losses, data
from ml.scripts.train_mnist_gan import train_step, get_train_context
from ml.scripts.train_mnist_gan_with_class_prompts import train_step, PROCESSING_OPS

logger = logging.getLogger("ML Tests.")
logger.setLevel(logging.DEBUG)


def test_rgb_image_generator_call():
    """
    Tests that the generators.RGBImageGenerator can be successfully initialized and called
    Returns:

    """
    batch_size = 10
    noise_dimension = 100
    output_image_size = 256

    generator = generators.RGBImageGenerator()
    sample_input = np.random.random((batch_size, noise_dimension))

    generated_images = generator(sample_input)
    assert generated_images.shape == (
        batch_size,
        output_image_size,
        output_image_size,
        3,
    )


def test_rgb_image_generator_output_sizes():
    """
    Tests that the generator outputs images of expected sizes
    Returns:

    """

    output_image_sizes = [64, 128, 256]
    batch_size = 10
    noise_dimension = 100
    sample_input = np.random.random((batch_size, noise_dimension))

    for image_size in output_image_sizes:
        generator = generators.RGBImageGenerator(output_image_size=image_size)
        generated_images = generator(sample_input)
        assert generated_images.shape == (batch_size, image_size, image_size, 3)


def test_mnist_gan_with_prompts_train_step():
    """Tests the train step for MNIST GAN with class label prompts

    Returns:

    """
    batch_size = 10
    noise_dimension = 100
    epochs = 1
    label_smoothing = False
    pre_processing = "unit_range"
    discriminator_noise = False

    generator_namespace = contexts.GeneratorNamespace(
        model=generators.RGBImageGeneratorWithTextPrompt(
            initial_filters=128,
            output_image_size=28,
            reshape_into=(7, 7, 256),
            embedding_dimension=7 * 7 * 256,
        ),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=losses.generator_loss,
    )

    # set up the discriminator
    discriminator_namespace = contexts.DiscriminatorNamespace(
        model=discriminators.ImageDiscriminator(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=losses.discriminator_loss,
    )

    context = contexts.MnistPromptGANContext(
        batch_size=batch_size,
        noise_dimension=noise_dimension,
        epochs=epochs,
        label_smoothing=label_smoothing,
        pre_processing=pre_processing,
        discriminator_noise=discriminator_noise,
        generator_namespace=generator_namespace,
        discriminator_namespace=discriminator_namespace,
    )

    # this will have 10 samples, instead of the number of batches
    reference = context.set_reference()
    assert len(reference) == 2, "Reference must contain two elements."
    processing_op = PROCESSING_OPS.get(context.pre_processing)
    dataset = data.get_mnist_dataset_with_labels(batch_size=context.batch_size, preprocess_fn=processing_op)

    generator_input_noise = context.generate_noise()
    true_images, labels = next(dataset.as_numpy_iterator())
    context.assign_inputs([generator_input_noise, true_images)
    step_loss = train_step(context)


def test_discriminator():
    """
    Tests that a discriminator can be succesfully initialized and called on a sample input
    Returns:

    """
    batch_size = 10
    image_size = 256
    output_dense_activation = "sigmoid"
    add_input_noise = True
    sample_input = np.random.random((batch_size, image_size, image_size, 3))
    discriminator = discriminators.ImageDiscriminator(
        output_dense_activation=output_dense_activation, add_input_noise=add_input_noise
    )

    sample_output = discriminator(sample_input)

    assert sample_output.shape == (
        batch_size,
        1,
    ), f"Output size isn't as expected {sample_output.shape}"
    assert (
        np.max(sample_output.numpy().ravel()) <= 1.0
    ), f"Max value in output array exceeds 1.0"
    assert (
        np.min(sample_output.numpy().ravel()) >= 0.0
    ), "Min value in output array is below 0.0"


def test_mnist_gan_train_step():
    batch_size = 8
    noise_dimension = 100
    epochs = 1

    context = get_train_context(batch_size, noise_dimension, epochs)

    # this will have 10 samples, instead of the number of batches
    reference = context.set_reference()
    syntehtic_images = np.random.uniform(0, 1.0, size=(batch_size, 28, 28, 1))
    generator_input_noise = context.generate_noise()
    context.assign_inputs(generator_input_noise, syntehtic_images)
    step_loss = train_step(context)


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
