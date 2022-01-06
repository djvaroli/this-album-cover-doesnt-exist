import logging

import tensorflow as tf
import numpy as np

from ml.model_components import generators, discriminators
from ml.training import contexts, losses, data
from ml.scripts import common as scripts_common
from ml.scripts import train_mnist_gan
from ml.scripts import train_conditional_mnist_gan


logger = logging.getLogger("ML Tests.")
logger.setLevel(logging.DEBUG)


def test_image_generator():
    """
    Tests that the generators.ImageGenerator can be successfully initialized and called
    Returns:

    """
    batch_size = 10
    noise_dimension = 100
    output_image_size = 256

    generator = generators.ImageGenerator(n_channels=3)
    sample_input = np.random.random((batch_size, noise_dimension))

    generated_images = generator(sample_input)
    assert generated_images.shape == (
        batch_size,
        output_image_size,
        output_image_size,
        3,
    )


def test_image_generator_output_sizes():
    """
    Tests that the generator outputs images of expected sizes
    Returns:

    """

    output_image_sizes = [64, 128, 256]
    batch_size = 10
    noise_dimension = 100
    sample_input = np.random.random((batch_size, noise_dimension))
    n_channels = 3

    for image_size in output_image_sizes:
        generator = generators.ImageGenerator(output_image_size=image_size, n_channels=n_channels)
        generated_images = generator(sample_input)
        assert generated_images.shape == (batch_size, image_size, image_size, n_channels)


def test_image_discriminator():
    """
    Tests that a discriminator can be succesfully initialized and called on a sample input
    Returns:

    """
    batch_size = 10
    image_size = 256
    add_input_noise = True
    output_dense_activation = "sigmoid"
    sample_input = np.random.random((batch_size, image_size, image_size, 3))
    discriminator = discriminators.ImageDiscriminator(
        output_dense_activation=output_dense_activation,
        add_input_noise=add_input_noise
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


def test_conditional_image_generator():
    batch_size = 10
    noise_dimension = 100
    output_image_size = 256
    n_channels = 1

    generator = generators.ConditionalImageGenerator(n_channels=n_channels)
    sample_input_noise = np.random.random((batch_size, noise_dimension))
    sample_input_prompt = np.random.random((batch_size, 10))

    generated_images = generator([sample_input_noise, sample_input_prompt])
    assert generated_images.shape == (
        batch_size,
        output_image_size,
        output_image_size,
        n_channels,
    )


def test_conditional_image_discriminator():
    batch_size = 1
    image_dimension = 28
    n_channels = 3
    input_image = tf.random.normal(shape=(batch_size, image_dimension, image_dimension, n_channels))
    input_labels = tf.random.normal(shape=(batch_size, 100))

    discriminator = discriminators.ConditionalImageDiscriminator()

    discriminator([input_image, input_labels])


def test_mnist_gan_train_step():
    batch_size = 8
    noise_dimension = 100
    epochs = 1

    context = train_mnist_gan.get_train_context(batch_size, noise_dimension, epochs)

    # this will have 10 samples, instead of the number of batches
    reference = context.set_reference()
    syntehtic_images = np.random.uniform(0, 1.0, size=(batch_size, 28, 28, 1))
    generator_input_noise = context.generate_noise()
    context.assign_inputs(generator_input_noise, syntehtic_images)
    step_loss = train_mnist_gan.train_step(context)


def test_conditional_mnist_gan_train_step():
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
        model=generators.ConditionalImageGenerator(
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
        model=discriminators.ConditionalImageDiscriminator(),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss_fn=losses.discriminator_loss,
    )

    context = contexts.ConditionalMNISTGANContext(
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

    processing_op = scripts_common.PROCESSING_OPS.get(context.pre_processing)
    dataset = data.get_mnist_dataset_with_labels(batch_size=context.batch_size, preprocess_fn=processing_op)

    generator_input_noise = context.generate_noise()
    noise_labels = context.generate_noise_labels()
    true_images, true_labels = next(dataset.as_numpy_iterator())

    context.assign_inputs([generator_input_noise, noise_labels], [true_images, true_labels])
    step_loss = train_conditional_mnist_gan.train_step(context)
