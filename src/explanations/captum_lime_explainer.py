import numpy as np

from typing import Callable
from captum.attr import visualization as viz
from captum.attr import Lime
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torchvision.transforms import transforms

from util import numpy_to_pil


def attribute_image_features(model, algorithm, input, target_label, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=target_label,
                                              **kwargs
                                              )

    return tensor_attributions


class CaptumLimeExplainer:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def explain(self, model, image, target_label, segments, classifier_func: Callable):
        lime = Lime(classifier_func)
        attr_lime = attribute_image_features(model,
                                             lime,
                                             image,
                                             target_label,
                                             feature_mask=segments,
                                             n_samples=1000,
                                             perturbations_per_eval=200)  # default of original LIME
        return attr_lime
        # pil_img = transforms.ToPILImage()(attr_lime.squeeze_(0))
        # attr_lime = np.transpose(attr_lime.squeeze().cpu().detach().numpy(), (1, 2, 0))

        # image = numpy_to_pil(attr_lime)
        # pil_img.show()
        # _ = viz.visualize_image_attr(image, image,
        #                              method="original_image", title="Original Image")
        # _ = viz.visualize_image_attr(attr_lime, image, sign="absolute_value",
        #                             title="Overlayed LIME")

        # return attr_lime
