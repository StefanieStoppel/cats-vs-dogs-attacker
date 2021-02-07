import numpy as np
import torch

from captum.attr import visualization as viz


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

    def visualize(self, attributions: torch.Tensor, images: torch.Tensor, titles=()):
        attr_dl = np.transpose(attributions.cpu().detach().numpy(), (0, 2, 3, 1))
        images_np = np.transpose((images.cpu().detach().numpy()), (0, 2, 3, 1))
        for idx, (img, attr) in enumerate(zip(images_np, attr_dl), 0):
            if len(titles) == 0:
                title_ = f"Attribution for {self.xai_algorithm.__class__.__name__} {idx + 1}"
            else:
                title_ = titles[idx]
            _ = viz.visualize_image_attr(attr,
                                         img,
                                         method="blended_heat_map",
                                         sign="all",
                                         show_colorbar=True,
                                         title=title_)
