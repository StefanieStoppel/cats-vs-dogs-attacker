import os

import torch
from captum.attr import DeepLift
from torch.utils.data import DataLoader

from pytorch_lightning import loggers as pl_loggers, Trainer
from torchvision import transforms

from config import DATA_ADV_PATH, LOGS_PATH, LOGS_ADV_PATH
from dataloader import DogVsCatWithAdversarialsDataset
from explanations.captum_explainer import CaptumExplainer
from losses.loss import similarity_loss_pcc, similarity_loss_ssim
from models.lit_fooled_model import LitFooledModel


def run_train_adv(config):
    lit_fooled_model = LitFooledModel(config["lr"])

    # Explainer
    explainer = CaptumExplainer(config["xai_algorithm"], lit_fooled_model.model)
    lit_fooled_model.set_explainer(explainer)

    train_dataset = DogVsCatWithAdversarialsDataset(config["train_csv"],
                                                    config["train_dir"],
                                                    transform=config["transform"])
    validation_dataset = DogVsCatWithAdversarialsDataset(config["validation_csv"],
                                                         config["validation_dir"],
                                                         transform=config["transform"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"])
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"],
                                   shuffle=False, num_workers=config["num_workers"])

    # init logging
    tb_logger = pl_loggers.TensorBoardLogger(config["logs_dir"])
    trainer = Trainer(logger=tb_logger, gpus=1)

    # run training
    trainer.fit(lit_fooled_model, train_loader, validation_loader)


def run_test_pcc(config):
    test_dataset = DogVsCatWithAdversarialsDataset(config["test_csv"],
                                                   config["test_dir"],
                                                   transform=config["transform"])

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"])

    original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = next(
        iter(test_loader))

    # 1) first model explanations
    lit_fooled_model = LitFooledModel(config["lr"]).load_from_checkpoint(
        checkpoint_path=config["checkpoint"]
    )
    explainer = CaptumExplainer(config["xai_algorithm"], lit_fooled_model.model)
    lit_fooled_model.set_explainer(explainer)
    first_original_explanation_maps, first_adversarial_explanation_maps = \
        lit_fooled_model._create_explanation_map(original_image, adversarial_image, gt_label, adv_label)

    # 2) second model explanations
    lit_fooled_model = LitFooledModel(config["lr"]).load_from_checkpoint(
        checkpoint_path=config["checkpoint_2"]
    )
    explainer = CaptumExplainer(config["xai_algorithm"], lit_fooled_model.model)
    lit_fooled_model.set_explainer(explainer)

    second_original_explanation_maps, second_adversarial_explanation_maps = \
        lit_fooled_model._create_explanation_map(original_image, adversarial_image, gt_label, adv_label)

    # calculate PCC losses
    first_pcc_loss = similarity_loss_ssim(first_original_explanation_maps, first_adversarial_explanation_maps)
    second_pcc_loss = similarity_loss_ssim(second_original_explanation_maps, second_adversarial_explanation_maps)
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

    # explainer.visualize(first_original_explanation_maps[0].unsqueeze(0), original_image[0].unsqueeze(0),
    #                     titles=("1) original", "1) adversarial"))
    # explainer.visualize(first_adversarial_explanation_maps[0].unsqueeze(0), original_image[0].unsqueeze(0),
    #                     titles=("1) original", "1) adversarial"))
    # explainer.visualize(second_original_explanation_maps[0].unsqueeze(0),
    #                     second_adversarial_explanation_maps[0].unsqueeze(0), titles=("2) original", "2) adversarial"))


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
        "checkpoint": os.path.join(LOGS_PATH, "default/version_10/checkpoints/epoch=0-step=136.ckpt"),
        "checkpoint_2": os.path.join(LOGS_ADV_PATH, "default/version_62/checkpoints/epoch=0-step=577.ckpt"),

        # Transform images
        "transform": transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),

        # Training
        "batch_size": 8,
        "num_workers": 2,
        "lr": 1e-4,

        # XAI algorithm
        "xai_algorithm": DeepLift,

    }
    # run_train_adv(config=CONFIG)
    run_test_pcc(config=CONFIG)
