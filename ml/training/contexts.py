from abc import ABC
import typing
import datetime
from datetime import datetime as dt

import tensorflow as tf
import numpy as np


class _ModelNamespace:
    def __init__(
        self,
        model: typing.Union[tf.keras.Model, tf.Module],
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: typing.Callable,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.__model_inputs = None

    @property
    def model_inputs(self):
        """
        Returns the next set of inputs to the model
        Returns:

        """
        return self.__model_inputs

    @model_inputs.setter
    def model_inputs(self, *args):
        raise Exception("Please use the assign_inputs function instead.")

    def assign_inputs(self, input_batch):
        """
        Assigns the next set of inputs to the model
        Args:
            input_batch:

        Returns:

        """
        self.__model_inputs = input_batch


class GeneratorNamespace(_ModelNamespace):
    """ """

    pass


class DiscriminatorNamespace(_ModelNamespace):
    """ """

    def __init__(
        self,
        model: typing.Union[tf.keras.Model, tf.Module],
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: typing.Callable,
    ):
        super(DiscriminatorNamespace, self).__init__(model, optimizer, loss_fn)
        self.image_batch = None

    def set_image_batch(self, image_batch: np.ndarray):
        """

        Args:
            image_batch:

        Returns:

        """
        self.image_batch = image_batch


class BaseModelTrainingContext:
    """[summary]"""

    def __init__(self, model_name: str, batch_size: int, epochs: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.date: datetime.datetime = dt.now()
        self.epochs = epochs


class BaseGANTrainingContext(BaseModelTrainingContext):
    """ """

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        noise_dimension: int,
        epochs: int,
        generator_namespace: GeneratorNamespace,
        discriminator_namespace: DiscriminatorNamespace,
    ):
        super(BaseGANTrainingContext, self).__init__(model_name, batch_size, epochs)
        self.noise_dimension = noise_dimension
        self.generator_namespace = generator_namespace
        self.discriminator_namespace = discriminator_namespace
        self.__reference = None

    def assign_inputs(
        self,
        generator_inputs: typing.Union[np.ndarray, tf.Tensor, typing.List[np.ndarray], typing.List[tf.Tensor]],
        discriminator_inputs: typing.Union[np.ndarray, tf.Tensor],
    ):
        """
        Sets the next batch of inputs for the generator and discriminator
        Args:
            generator_inputs:
            discriminator_inputs:

        Returns:

        """
        self.generator_namespace.assign_inputs(generator_inputs)
        self.discriminator_namespace.assign_inputs(discriminator_inputs)
        return self

    def generate_noise(
        self, mean: float = 0.0, stddev: float = 1.0, **kwargs
    ) -> tf.Tensor:
        """
        Generates a tensor of samples of shape (BATCH_SIZE, NOISE_DIMENSION), where each entry is sampled
        from a Gaussian distribution.
        Returns:

        """
        return tf.random.normal(
            (self.batch_size, self.noise_dimension), mean, stddev, **kwargs
        )

    @property
    def reference(self):
        """
        Returns a pre-generated tensor of Gaussian-distributed noise to be used as a reference when
        evaluating the GAN
        Returns:

        """
        return self.__reference

    @reference.setter
    def reference(self, *args):
        """
        The reference noise cannot be set as an attribute directly, but must be set using the
        set_reference method.
        Args:
            *args:

        Returns:

        """
        raise Exception(
            "Reference cannot be set directly, please use BaseGANTrainingContext.set_reference method."
        )

    def set_reference(
        self, size: int = 10, mean: float = 0.0, stddev: float = 1.0, **kwargs
    ):
        """Sets the reference noise to be used when evaluating the GAN

        Args:
            size:
            mean:
            stddev:
            **kwargs:

        Returns:

        """
        if self.reference is None:
            self.__reference = tf.random.normal(
                (size, self.noise_dimension), mean, stddev, **kwargs
            )
        else:
            raise Exception(
                "Reference already set, please clear reference using BaseGANTrainingContext.clear_reference first."
            )

        return self.reference

    def clear_reference(self):
        """
        Clears an existing reference
        Returns:

        """

        self.__reference = None

        return self


class MNISTGANContext(BaseGANTrainingContext):
    """[summary]

    Args:
        BaseModelTrainingContext ([type]): [description]
    """

    model_name = "mnist-gan"

    def __init__(
        self,
        batch_size: int,
        noise_dimension: int,
        epochs: int,
        generator_namespace: GeneratorNamespace,
        discriminator_namespace: DiscriminatorNamespace,
        label_smoothing: bool = False,
        discriminator_noise: bool = False,
        pre_processing: str = "unit_range",
    ):
        super(MNISTGANContext, self).__init__(
            self.model_name,
            batch_size,
            noise_dimension,
            epochs,
            generator_namespace,
            discriminator_namespace,
        )
        self.label_smoothing = label_smoothing
        self.discriminator_noise = discriminator_noise
        self.pre_processing = pre_processing


class MnistPromptGANContext(BaseGANTrainingContext):
    """[summary]

    Args:
        MnistPromptGANContext ([type]): [description]
    """

    model_name = "mnist-gan-with-prompts"

    def __init__(
        self,
        batch_size: int,
        noise_dimension: int,
        epochs: int,
        generator_namespace: GeneratorNamespace,
        discriminator_namespace: DiscriminatorNamespace,
        label_smoothing: bool = False,
        discriminator_noise: bool = False,
        pre_processing: str = "unit_range",
    ):
        super(MnistPromptGANContext, self).__init__(
            self.model_name,
            batch_size,
            noise_dimension,
            epochs,
            generator_namespace,
            discriminator_namespace,
        )
        self.label_smoothing = label_smoothing
        self.discriminator_noise = discriminator_noise
        self.pre_processing = pre_processing

    def set_reference(
        self, size: int = 10, mean: float = 0.0, stddev: float = 1.0, **kwargs
    ) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        if self.reference is None:
            reference_noise = tf.random.normal(
                (size, self.noise_dimension), mean, stddev, **kwargs
            )
            labels = tf.expand_dims(tf.cast(tf.linspace(0, 9, num=10), tf.int16), axis=0)
            reference_labels = tf.tile(labels, multiples=[size, 1])
            self.__reference = [reference_noise, reference_labels]
        else:
            raise Exception(
                "Reference already set, please clear reference using BaseGANTrainingContext.clear_reference first."
            )

        return self.reference
