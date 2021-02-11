import io

import matplotlib.pyplot as plt
from functools import wraps
from time import time

import numpy as np
from PIL import Image
from numpy import asarray
from torch import nn


def timeit(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f"function {func.__name__} took {1000*(te-ts):.1f} ms")
        return result
    return wrap


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def numpy_to_pil(np_array):
    image = Image.fromarray(np_array)
    return image


def pil_read(image_path: str):
    return Image.open(image_path)


def load_image_as_numpy_array(image_path: str):
    image = pil_read(image_path)
    return asarray(image)


def rgb_tensor_to_pil_numpy(rgb_tensor):
    return np.transpose(rgb_tensor.cpu().detach().numpy(), (0, 2, 3, 1))


def map_image_to_unit_interval(image):
    # add 1 to map image in range [-1, 1] to [0, 2]
    tensor_image = image + 1
    # step 2: convert it to [0 ,1]
    tensor_image = tensor_image - tensor_image.min()
    tensor_image_0_1 = tensor_image / (tensor_image.max() - tensor_image.min())
    return tensor_image_0_1


def map_explanations_forward(explanations):
    return (explanations + 1) / 2


def map_explanations_backward(explanations):
    return (explanations * 2) - 1

