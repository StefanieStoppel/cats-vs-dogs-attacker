import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from functools import partial
from typing import Callable, Tuple
from captum.attr import DeepLift
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torchvision.io import read_image
from torchvision.transforms import transforms

from config import LOGS_PATH
from explanations.captum_explainer import CaptumExplainer
from explanations.captum_lime_explainer import CaptumLimeExplainer
from explanations.lime_explainer import LimeExplainer
from lit_model import LitVGG16Model
from util import load_image_as_numpy_array, pil_read, timeit, inverse_normalize


def explain(explainer, model, classifier_function: Callable, original_image: np.array, adversarial_image: np.array):
    # put model in evaluation mode
    model.eval()

    orig_boundary = explainer.explain(original_image, classifier_func=classifier_function)
    adv_boundary = explainer.explain(adversarial_image, classifier_func=classifier_function)

    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(orig_boundary)
    # ax[1].imshow(adv_boundary)
    # plt.show()


@timeit
def explain_captum_lime(model,
                        classifier_function: Callable,
                        images: torch.Tensor,
                        labels: torch.Tensor,
                        segments: np.array):
    model.eval()

    captum_lime_exlainer = CaptumLimeExplainer(random_seed=CONFIG["random_seed"])
    orig_expl = captum_lime_exlainer.explain(model,
                                             images,
                                             labels,
                                             segments,
                                             classifier_func=classifier_function)
    # adv_expl = captum_lime_exlainer.explain(model,
    #                                         images[1].unsqueeze(0),
    #                                         labels[1],
    #                                         segments[1],
    #                                         classifier_func=classifier_function)


def explain_captum_deeplift(explainer: CaptumExplainer,
                            model: torch.nn.Module,
                            images: torch.Tensor,
                            labels: torch.Tensor,
                            titles: Tuple):
    attr_dl = explainer.explain(model, images, labels, baselines=images * 0)
    explainer.visualize(attr_dl, images, titles=titles)


def classify_dogs_vs_cats(model, device, images_np: np.array):
    images = torch.stack(tuple(transforms.ToTensor()(i) for i in images_np), dim=0).to(device)
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def classify_function(model, images):
    logits = model(images)
    # probs = F.softmax(logits, dim=1)
    return logits


if __name__ == '__main__':
    CONFIG = {
        # Paths
        "original_image_path": "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/adversarials/LinfFastGradientAttack/0.005/cat.11489_orig.jpg",
        "adversarial_image_path": "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/adversarials/LinfFastGradientAttack/0.005/cat.11489_adv.jpg",
        "checkpoint": os.path.join(LOGS_PATH, "default/version_10/checkpoints/epoch=0-step=136.ckpt"),

        # other
        "random_seed": 42,
        "transform": transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
        "use_cuda": True
    }

    # GPU or CPU
    device = torch.device('cuda' if (torch.cuda.is_available() and CONFIG["use_cuda"]) else 'cpu')

    # Load model
    lit_model = LitVGG16Model.load_from_checkpoint(checkpoint_path=CONFIG["checkpoint"])
    # model = torchvision.models.vgg16(pretrained=True)
    model = lit_model.model
    model = model.to(device)

    # Explainer
    # lime_explainer = LimeExplainer(random_seed=CONFIG["random_seed"])
    deeplift_explainer = CaptumExplainer(DeepLift, model)

    # Load images as numpy arrays
    original_img_np = load_image_as_numpy_array(CONFIG["original_image_path"])
    adversarial_img_np = load_image_as_numpy_array(CONFIG["adversarial_image_path"])

    # create segments
    segmentation_function = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                  max_dist=200, ratio=0.2,
                                                  random_seed=CONFIG["random_seed"])
    original_segments = segmentation_function(original_img_np)
    adversarial_segments = segmentation_function(adversarial_img_np)
    segments = torch.from_numpy(np.stack((original_segments, adversarial_segments))).to(device)

    # Load images as PyTorch Tensors
    transform = CONFIG["transform"]
    original_img_tensor = transform(pil_read(CONFIG["original_image_path"]))
    adversarial_img_tensor = transform(pil_read(CONFIG["adversarial_image_path"]))
    images_tensor = torch.stack((original_img_tensor, adversarial_img_tensor), dim=0).to(device)
    # images_tensor = (images_tensor / 255.0).to(device)
    # labels_tensor = torch.tensor((990, 73)).to(device)

    # cat: 0; dog: 1
    labels_tensor = torch.tensor((0, 1)).to(device)
    plot_titles = ("DeepLIFT for cat (0, original)", "DeepLIFT for dog (1, adversarial)")
    # labels_tensor = [1, 0]

    # Create classifier function
    classifier_func = partial(classify_dogs_vs_cats, model, device)
    # classifier_func = partial(classify_function, model)

    explain_captum_deeplift(deeplift_explainer, model, images_tensor, labels_tensor, titles=plot_titles)
    # explain(lime_explainer, model, classifier_func, original_img_np, adversarial_img_np)
    # explain_captum_lime(model, classifier_func, images_tensor, labels_tensor, segments)
