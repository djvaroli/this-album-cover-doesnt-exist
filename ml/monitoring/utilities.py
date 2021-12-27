
# from PIL import Image

import numpy as np

class Image:
    pass


def grid_from_array():
    """

    Returns:

    """
    pass


def array_to_image(arr: np.ndarray) -> Image:
    """Given a numpy array tries to convert into an image

    Args:
        arr (np.ndarray): A numpy array that can be converted into an image

    Returns:
        Image: An instance of PIL.Image class obtained from the input array
    """
    
    return Image.fromarray(arr)


def save_image(img: Image, filepath: str):
    """Saves an image in a desired location

    Args:
        img (Image): The image to be saved
        filepath (str): The destination, i.e. where to save the image
    """
    img.save(filepath)