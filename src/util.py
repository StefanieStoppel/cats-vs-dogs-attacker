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


def load_image_as_numpy_array(image_path: str):
    image = Image.open(image_path)
    return asarray(image)
