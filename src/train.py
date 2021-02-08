import os

from pytorch_lightning import loggers as pl_loggers, Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from config import DATA_PATH, LOGS_PATH
from dataloader import get_dogs_vs_cats_data_splits
from models.lit_model import LitVGG16Model


def run(config):
    train_dir = config["train_dir"]
    logs_dir = config["logs_dir"]
    data_splits = config["data_split"]
    num_workers = config["num_workers"]
    data_transforms = config["transform"]
    hparams = config["hparams"]
    batch_size = hparams["batch_size"]

    lit_vgg16_model = LitVGG16Model(hparams)

    train_dataset, validation_dataset, test_dataset = get_dogs_vs_cats_data_splits(
        train_dir,
        data_splits,
        transform=data_transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

    # init logging
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
    trainer = Trainer(logger=tb_logger, gpus=1)

    # run training
    trainer.fit(lit_vgg16_model, train_loader, validation_loader)


def run_test(config):
    lit_vgg16_model = LitVGG16Model.load_from_checkpoint(
        checkpoint_path=config["checkpoint"]
    )
    train_dataset, validation_dataset, test_dataset = get_dogs_vs_cats_data_splits(
        config["train_dir"],
        config["data_split"],
        transform=config["transform"]
    )

    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"])

    tb_logger = pl_loggers.TensorBoardLogger(config["logs_dir"])
    trainer = Trainer(logger=tb_logger, gpus=1)

    trainer.test(model=lit_vgg16_model, test_dataloaders=test_loader)


if __name__ == '__main__':
    CONFIG = {
        # Directories
        "train_dir": os.path.join(DATA_PATH, "train"),
        "test_dir": os.path.join(DATA_PATH, "test"),
        "train_csv": os.path.join(DATA_PATH, "train_list.csv"),
        "test_csv": os.path.join(DATA_PATH, "test_list.csv"),
        "logs_dir": LOGS_PATH,
        "checkpoint": os.path.join(LOGS_PATH, "default/version_10/checkpoints/epoch=0-step=136.ckpt"),
        # "checkpoint": "best",

        # Transform images
        "transform": transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
        "hparams": {
            "lr": 1e-4,
            "batch_size": 128,
        },
        # Training
        "data_split": (0.7, 0.2, 0.1),
        "num_workers": 8,
    }
    run_test(CONFIG)
    # run(CONFIG)
