import os

from pytorch_lightning import loggers as pl_loggers, Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import DATA_PATH
from dataloader import get_dogs_vs_cats_data_splits
from lit_model import LitVGG16Model
from src import ROOT_DIR


def run(config):
    lit_vgg16_model = LitVGG16Model(config["lr"])

    train_dataset, validation_dataset, test_dataset = get_dogs_vs_cats_data_splits(
        config["train_dir"],
        config["data_splits"],
        transform=config["transform"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"])
    validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"],
                                   shuffle=False, num_workers=config["num_workers"])

    # init logging
    tb_logger = pl_loggers.TensorBoardLogger(config["logs_dir"])
    trainer = Trainer(logger=tb_logger, gpus=1)

    # run training
    trainer.fit(lit_vgg16_model, train_loader, validation_loader)


if __name__ == '__main__':
    CONFIG = {
        # Directories
        "train_dir": os.path.join(DATA_PATH, "train"),
        "test_dir": os.path.join(DATA_PATH, "test"),
        "train_csv": os.path.join(DATA_PATH, "train_list.csv"),
        "test_csv": os.path.join(DATA_PATH, "test_list.csv"),
        "logs_dir": os.path.join(ROOT_DIR, "logs"),

        # Transform images
        "transform": transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),

        # Training
        "data_splits": (0.7, 0.2, 0.1),
        "batch_size": 128,
        "num_workers": 8,
        "lr": 1e-3
    }
    run(CONFIG)
