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
            loss_fn: typing.Callable
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn


class GeneratorNamespace(_ModelNamespace):
    """

    """
    pass


class DiscriminatorNamespace(_ModelNamespace):
    """

    """

    def __init__(
            self,
            model: typing.Union[tf.keras.Model, tf.Module],
            optimizer: tf.keras.optimizers.Optimizer,
            loss_fn: typing.Callable
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
    """[summary]
    """
    def __init__(
            self,
            model_name: str,
    ):
        self.model = model_name
        self.model_inputs = None
        self.date: datetime.datetime = dt.now()


class MNISTGANContext(BaseModelTrainingContext):
    """[summary]

    Args:
        BaseModelTrainingContext ([type]): [description]
    """
    model_name = "mnist-gan-context"

    def __init__(
            self,
            generator_namespace: GeneratorNamespace,
            discriminator_namespace: DiscriminatorNamespace
    ):
        super(MNISTGANContext, self).__init__(self.model_name)
        self.generator_namespace = generator_namespace
        self.discriminator_namespace = discriminator_namespace

    def set_discriminator_images(self, images: np.ndarray):
        """
        Sets the images to be used as input to the discriminator
        Args:
            images:

        Returns:

        """
        self.discriminator_namespace.set_image_batch(images)
        return self


