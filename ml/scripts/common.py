import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop


PROCESSING_OPS = {
    "normalize": lambda x: x / 255.0,
    "unit_range": lambda x: (x - 127.5) / 127.5,
}

PREPROCESSING_OP_ACTIVATION = {
    "normalize": "sigmoid",
    "unit_range": "tanh"
}

OPTIMIZERS = {
    "adam": Adam,
    "rmsprop": RMSprop
}