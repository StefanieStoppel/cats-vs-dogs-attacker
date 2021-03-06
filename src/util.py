from functools import wraps
from time import time
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
