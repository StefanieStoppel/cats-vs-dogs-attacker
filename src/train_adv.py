import os

from captum.attr import DeepLift
from torch.utils.data import DataLoader

from pytorch_lightning import loggers as pl_loggers, Trainer
from torchvision import transforms

from config import DATA_ADV_PATH, LOGS_PATH, LOGS_ADV_PATH
from dataloader import DogVsCatWithAdversarialsDataset
from explanations.captum_explainer import CaptumExplainer
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

        # Transform images
        "transform": transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),

        # Training
        "batch_size": 10,
        "num_workers": 2,
        "lr": 1e-2,

        # XAI algorithm
        "xai_algorithm": DeepLift,

    }
    run_train_adv(config=CONFIG)
