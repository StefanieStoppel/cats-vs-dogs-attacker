from functools import partial
from typing import Union, Tuple

import numpy as np
import torch

from captum.attr import visualization as viz
from matplotlib.pyplot import axis, figure
from util import rgb_tensor_to_pil_numpy


class CaptumExplainer:
    def __init__(self, algorithm, random_seed=42):
        self.xai_algorithm = algorithm
        self.random_seed = random_seed

    def attribute_image_features(self, image, target_label, **kwargs):
        tensor_attributions = self.xai_algorithm.attribute(image,
                                                           target=target_label,
                                                           **kwargs
                                                           )
        return tensor_attributions

    def explain(self, image, target_label, **kwargs):
        explanation = self.attribute_image_features(image,
                                                    target_label,
                                                    **kwargs)
        return explanation

    def visualize(self,
                  attributions: torch.Tensor,
                  images: torch.Tensor,
                  fig_ax_tuple: Union[None, Tuple[figure, axis]] = None,
                  titles: Tuple = ()
                  ):
        attr_dl = rgb_tensor_to_pil_numpy(attributions)
        images_np = rgb_tensor_to_pil_numpy(images)
        for idx, (img, attr) in enumerate(zip(images_np, attr_dl), 0):
            if len(titles) == 0:
                title_ = f"Attribution for {self.xai_algorithm.__class__.__name__} {idx + 1}"
            else:
                title_ = titles[idx]
            self.visualize_single(attr, img, fig_ax_tuple, title_)

    def visualize_single(self,
                         attribution: np.ndarray,
                         image: np.ndarray,
                         fig_ax_tuple: Union[None, Tuple[figure, axis]] = None,
                         title: str = None):
        return viz.visualize_image_attr(attribution,
                                        image,
                                        plt_fig_axis=fig_ax_tuple,
                                        method="blended_heat_map",
                                        sign="all",
                                        show_colorbar=True,
                                        title=title)


def get_explainer(xai_algorithm, lit_fooled_model, **kwargs):
    algorithm_name = xai_algorithm.__name__
    if algorithm_name == "DeepLift":
        explainer = CaptumExplainer(xai_algorithm(lit_fooled_model.model))
    if algorithm_name == "LayerGradCam":
        last_conv2d = lit_fooled_model.model.features[28]
        explainer = CaptumExplainer(xai_algorithm(lit_fooled_model.model.forward, last_conv2d))
    explainer.explain = partial(explainer.explain, **kwargs)
    return explainer
