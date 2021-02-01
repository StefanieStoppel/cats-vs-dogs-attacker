from typing import Callable

import numpy as np
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

from config import LABEL_MAPPING


class LimeExplainer:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.explainer = LimeImageExplainer(random_state=self.random_seed)

    def explain(self, image_array: np.array, classifier_func: Callable, top_labels: int = 2):
        """
        Image (224x224) needs to be PyTorch Tensor.
        :return:
        """
        explanation = self.explainer.explain_instance(image_array,
                                                      classifier_func,
                                                      top_labels=top_labels,
                                                      hide_color=0,
                                                      random_seed=self.random_seed,
                                                      num_samples=1000)
        img, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                   positive_only=False,
                                                   num_features=10,
                                                   hide_rest=False)

        probabilities = classifier_func([image_array])

        print(f"Top labels:")
        for idx, (class_idx, class_probability) in enumerate(zip(explanation.top_labels, probabilities), 1):
            print(f"\t{class_idx})  '{LABEL_MAPPING[str(class_idx)]}: {class_probability.max()}'")

        img_boundary = mark_boundaries(img / 255.0, mask)
        return img_boundary

