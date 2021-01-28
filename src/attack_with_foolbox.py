from typing import Union, Sequence

import eagerpy as ep
import foolbox as fb
import numpy as np
import torch
import torchvision.models as models

from eagerpy import Tensor

from src.config import LABEL_MAPPING
from src.utils.general import timeit

CONFIG = {
    "epsilons": np.linspace(0.001, 0.005, num=1),
    "bounds": (0, 1),
    "preprocessing": dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3),
    "attack": fb.attacks.LinfFastGradientAttack(),
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
    for i, (eps, acc) in enumerate(zip(epsilons, robust_accuracy)):
        print(f"!!!! Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        adversarials = [j for j, adv in enumerate(is_adv[i]) if adv == True]
        for adv_idx in adversarials:
            img = images[adv_idx]
            adv_img = clipped[i][adv_idx]
            print(f"Ground truth label: '{labels[adv_idx]}'")

            original_label_id = foolbox_model(img.unsqueeze(0)).argmax().item()
            adv_label_id = foolbox_model(adv_img.unsqueeze(0)).argmax().item()
            print(f"Original prediction: {original_label_id}")
            print(f"Adversarial prediction: {adv_label_id}")
            print("")

        # first_adv_idx = np.random.choice(adversarials)
        # torchvision.utils.save_image(images[first_adv_idx],
        #                              os.path.join(get_root(), f"test_images/{first_adv_idx}_orig_eps_{eps}.jpg"))
        # torchvision.utils.save_image(clipped[i][first_adv_idx],
        #                              os.path.join(get_root(), f"test_images/{first_adv_idx}_adv_eps_{eps}.jpg"))


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(pretrained=True)
    # model = model.to(device)
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=CONFIG["bounds"], preprocessing=CONFIG["preprocessing"])
    images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=20)

    attack = CONFIG["attack"]

    run_attack(attack, fmodel, images, labels, CONFIG)