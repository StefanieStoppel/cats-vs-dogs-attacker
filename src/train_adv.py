import os

import torch
from captum.attr import DeepLift
from torch.utils.data import DataLoader
from operator import itemgetter
from pytorch_lightning import loggers as pl_loggers, Trainer
from torchvision import transforms

from config import DATA_ADV_PATH, LOGS_PATH, LOGS_ADV_PATH
from dataloader import DogVsCatWithAdversarialsDataset
from explanations.captum_explainer import CaptumExplainer
from losses.loss import similarity_loss_ssim
from models.lit_fooled_model import LitFooledModel


def run_train_adv(config):
    train_dir, train_csv, validation_dir, validation_csv, \
    logs_dir, num_workers, data_transforms, xai_algorithm, hparams = itemgetter("train_dir",
                                                                                "train_csv",
                                                                                "validation_dir",
                                                                                "validation_csv",
                                                                                "logs_dir",
                                                                                "num_workers",
                                                                                "transform",
                                                                                "xai_algorithm",
                                                                                "hparams")(config)
    batch_size = hparams["batch_size"]

    lit_fooled_model = LitFooledModel(hparams)

    # Explainer
    explainer = CaptumExplainer(xai_algorithm(lit_fooled_model.model))
    lit_fooled_model.set_explainer(explainer)

    train_dataset = DogVsCatWithAdversarialsDataset(train_csv,
                                                    train_dir,
                                                    transform=data_transforms)
    validation_dataset = DogVsCatWithAdversarialsDataset(validation_csv,
                                                         validation_dir,
                                                         transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

    # init logging
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
    trainer = Trainer(logger=tb_logger, gpus=1)

    # run training
    trainer.fit(lit_fooled_model, train_loader, validation_loader)


def run_test_pcc(config):
    test_dir, test_csv, num_workers, data_transforms, xai_algorithm, checkpoint, checkpoint2, hparams = \
        itemgetter(
            "test_dir"
            "test_csv"
            "train_csv"
            "num_workers"
            "transform"
            "xai_algorithm"
            "checkpoint"
            "checkpoint2"
            "hparams")(config)

    batch_size = hparams["batch_size"]

    test_dataset = DogVsCatWithAdversarialsDataset(test_csv,
                                                   test_dir,
                                                   transform=data_transforms)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = next(
        iter(test_loader))

    first_orig_sim = similarity_loss_ssim(original_image, adversarial_image)

    # 1) first model explanations
    lit_fooled_model = LitFooledModel(hparams).load_from_checkpoint(
        checkpoint_path=checkpoint
    )
    explainer = CaptumExplainer(xai_algorithm(lit_fooled_model.model))
    lit_fooled_model.set_explainer(explainer)
    first_original_explanation_maps, first_adversarial_explanation_maps = \
        lit_fooled_model._create_explanation_map(original_image, adversarial_image, gt_label, adv_label)

    # 2) second model explanations
    lit_fooled_model = LitFooledModel(hparams).load_from_checkpoint(
        checkpoint_path=checkpoint2
    )
    explainer = CaptumExplainer(xai_algorithm(lit_fooled_model.model))
    lit_fooled_model.set_explainer(explainer)

    second_original_explanation_maps, second_adversarial_explanation_maps = \
        lit_fooled_model._create_explanation_map(original_image, adversarial_image, gt_label, adv_label)

    # calculate PCC losses
    first_pcc_loss = similarity_loss_ssim(first_original_explanation_maps, first_adversarial_explanation_maps)
    second_pcc_loss = similarity_loss_ssim(second_original_explanation_maps, second_adversarial_explanation_maps)
    print(f"first first_orig_sim: {first_orig_sim}")
    print(f"first ssim_loss: {first_pcc_loss}")
    print(f"second ssim_loss: {second_pcc_loss}")

    print(f"visualizations for image {original_image_name[0]}.")
    attributions = torch.stack((first_original_explanation_maps[0],
                                second_original_explanation_maps[0],
                                first_adversarial_explanation_maps[0],
                                second_adversarial_explanation_maps[0]),
                               dim=0)
    images = torch.stack((original_image[0],
                          original_image[0],
                          original_image[0],
                          original_image[0],
                          ),
                         dim=0)

    explainer.visualize(attributions, images,
                        titles=("1) original", "2) original", "1) adv", "2) adv"))


if __name__ == '__main__':
    CONFIG = {
        # Directories
        "train_dir": os.path.join(DATA_ADV_PATH, "train"),
        "validation_dir": os.path.join(DATA_ADV_PATH, "validation"),
        "test_dir": os.path.join(DATA_ADV_PATH, "test"),

        "train_csv": os.path.join(DATA_ADV_PATH, "train_adv.csv"),
        "validation_csv": os.path.join(DATA_ADV_PATH, "validation_adv.csv"),
        "test_csv": os.path.join(DATA_ADV_PATH, "test_adv.csv"),

        "logs_dir": LOGS_ADV_PATH,
        # checkpoint of unmodified cat-dog classifier
        "checkpoint": os.path.join(LOGS_PATH, "default/version_13/checkpoints/epoch=0-step=136.ckpt"),
        # checkpoint of adversarially trained cat-dog classifier
        "checkpoint_2": os.path.join(LOGS_ADV_PATH, "default/version_73/checkpoints/epoch=1-step=1982.ckpt"),

        # Transform images
        "transform": transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),

        # Training
        "hparams": {
            "lr": 1e-4,
            "batch_size": 8,
            "num_classes": 2,
            "gammas": (1, 1, 10),
            "xai_algorithm": DeepLift,
            "similarity_loss": similarity_loss_ssim,
        },
        "num_workers": 2,
    }
    run_train_adv(config=CONFIG)
    # run_test_pcc(config=CONFIG)
