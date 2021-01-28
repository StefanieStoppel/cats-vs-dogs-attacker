import os
import pprint
import eagerpy as ep
import foolbox as fb
import numpy as np
import torch
import torchvision
import torchvision.models as models

from typing import Union, Sequence
from eagerpy import Tensor

from src.config import DATA_PATH
from src.util.general import timeit

pp = pprint.PrettyPrinter(indent=4, width=120)

CONFIG = {
    "epsilons": np.linspace(0.001, 0.005, num=1),
    "bounds": (0, 1),
    "preprocessing": dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3),
    "attack": fb.attacks.LinfFastGradientAttack(),
    "save_adversaries": True,
    "target_dir": os.path.join(DATA_PATH, "adversarials")
}


def wrap_model(model, bounds, preprocessing):
    foolbox_model = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
    return foolbox_model


@timeit
def attack_model(attack: fb.Attack,
                 foolbox_model: fb.PyTorchModel,
                 images: ep.Tensor,
                 labels: ep.Tensor,
                 epsilons: Union[Sequence[Union[float, None]], float, None]):
    return attack(foolbox_model, images, labels, epsilons=epsilons)


def filter_correctly_classified(foolbox_model: fb.PyTorchModel, images: Tensor, labels: ep.Tensor):
    correctly_classified_mask = get_correctly_classified_mask(foolbox_model, images, labels)
    return images[correctly_classified_mask], labels[correctly_classified_mask]


def get_correctly_classified_mask(foolbox_model: fb.PyTorchModel, images: ep.Tensor, labels: ep.Tensor):
    """
    Checks whether the model predicts the "correct", ground-truth labels for the (non-adversarial) input images.
    :param model: foolbox model
    :param images: tensor of non-adversarial images (B, C, W, H)
    :param labels: tensor of ground truth labels for the images (B, N), where N = number of classes
    :return: boolean tensor containing True for images that were correctly classified by the model, and False otherwise
    """
    images_, restore_type_images = ep.astensor_(images)
    labels_, restore_type_labels = ep.astensor_(labels)

    predictions = foolbox_model(images_).argmax(axis=-1)
    return restore_type_images(predictions == labels_)


def run_attack(fb_attack: fb.Attack, foolbox_model: fb.PyTorchModel, images: ep.Tensor, labels: ep.Tensor, config):
    print(f"Initial accuracy on images: {fb.utils.accuracy(foolbox_model, images, labels)}")

    # filter input images and labels so that only the ones which are correctly classified are kept
    print(f"before filtering: {images.shape}; {labels.shape}")
    images, labels = filter_correctly_classified(foolbox_model, images, labels)
    print(f"after filtering: {images.shape}; {labels.shape}")

    epsilons = config["epsilons"]
    raw, clipped, is_adv = attack_model(fb_attack, foolbox_model, images, labels, epsilons)

    robust_accuracy = 1 - ep.astensor(is_adv.type(torch.FloatTensor)).mean(axis=-1)

    print("Predictions and robust accuracies: ")
    for i, (eps, acc, clipped_adv) in enumerate(zip(epsilons, robust_accuracy, clipped)):
        print(f"!!!! Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        adversarial_mask = torch.nonzero(is_adv.flatten()).flatten()
        original_images = images[adversarial_mask]
        adversarial_images = clipped_adv[adversarial_mask]

        original_pred = foolbox_model(original_images).argmax(axis=-1)
        adv_pred = foolbox_model(adversarial_images).argmax(axis=-1)
        ground_truth_labels = labels[adversarial_mask]

        pp.pprint(f"Ground truth: {ground_truth_labels}")
        pp.pprint(f"Original:     {original_pred}")
        pp.pprint(f"Adversarial:  {adv_pred}")

        if config["save_adversaries"]:
            target_dir = os.path.join(config["target_dir"], fb_attack.__class__.__name__)
            os.makedirs(target_dir, exist_ok=True)
            print(f"Saving original & adversarial images in directory '{target_dir}'.")

            for original_img, adv_img, image_index in zip(original_images, adversarial_images, adversarial_mask):
                original_path = os.path.join(target_dir, f"{image_index}_orig_eps_{eps}.jpg")
                adv_path = os.path.join(target_dir, f"{image_index}_adv_eps_{eps}.jpg")
                torchvision.utils.save_image(original_img, original_path)
                torchvision.utils.save_image(adv_img, adv_path)


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(pretrained=True)
    # model = model.to(device)
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=CONFIG["bounds"], preprocessing=CONFIG["preprocessing"])
    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=20)

    attack = CONFIG["attack"]

    run_attack(attack, fmodel, images, labels, CONFIG)
