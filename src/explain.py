import os
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torchvision.io import read_image
from torchvision.transforms import transforms

from config import LOGS_PATH
from explanations.lime_explainer import LimeExplainer
from lit_model import LitVGG16Model
from util import load_image_as_numpy_array


def explain(explainer, model, classifier_function: Callable, original_image: np.array, adversarial_image: np.array):
    # put model in evaluation mode
    model.eval()

    orig_boundary = explainer.explain(original_image, classifier_func=classifier_function)
    adv_boundary = explainer.explain(adversarial_image, classifier_func=classifier_function)

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(orig_boundary)
    ax[1].imshow(adv_boundary)
    plt.show()


def classify_dogs_vs_cats(model, device, images_np: np.array):
    images = torch.stack(tuple(transforms.ToTensor()(i) for i in images_np), dim=0).to(device)
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


if __name__ == '__main__':
    CONFIG = {
        # Paths
        "original_image_path": "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/"
                               "adversarials/LinfFastGradientAttack/0.005/cat.55_orig.jpg",
        "adversarial_image_path": "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/"
                                  "adversarials/LinfFastGradientAttack/0.005/cat.55_adv.jpg",
        "checkpoint": os.path.join(LOGS_PATH, "default/version_1/checkpoints/epoch=0-step=136.ckpt"),


        # other
        "random_seed": 42,
        "transform": transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
        "use_cuda": True
    }

    # GPU or CPU
    device = torch.device('cuda' if (torch.cuda.is_available() and CONFIG["use_cuda"]) else 'cpu')

    # Load model
    lit_model = LitVGG16Model.load_from_checkpoint(checkpoint_path=CONFIG["checkpoint"])
    model = lit_model.model
    model = model.to(device)

    # Explainer
    lime_explainer = LimeExplainer(random_seed=CONFIG["random_seed"])

    # Load images as numpy arrays
    original_img_np = load_image_as_numpy_array(CONFIG["original_image_path"])
    adversarial_img_np = load_image_as_numpy_array(CONFIG["adversarial_image_path"])

    # Load images as PyTorch Tensors
    original_img_tensor = read_image(CONFIG["original_image_path"]).to(device)
    adversarial_img_tensor = read_image(CONFIG["adversarial_image_path"]).to(device)
    images_tensor = torch.cat((original_img_tensor, adversarial_img_tensor), 0)

    # Create classifier function
    classifier_func = partial(classify_dogs_vs_cats, model, device)

    explain(lime_explainer, model, classifier_func, original_img_np, adversarial_img_np)
