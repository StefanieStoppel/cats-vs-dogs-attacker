import torch
from torch.utils.data import DataLoader
from torchvision import datasets


def get_dogs_vs_cats_data_splits(data_dir, data_split=(0.7, 0.2, 0.1), transform=None, random_seed=42):
    dogs_vs_cats_dataset = datasets.ImageFolder(data_dir, transform=transform)

    dataset_len = len(dogs_vs_cats_dataset)
    train_val_split = [
        int(data_split[0] * dataset_len),
        int(data_split[1] * dataset_len),
        int(data_split[2] * dataset_len),
    ]
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dogs_vs_cats_dataset,
                                                                                    train_val_split,
                                                                                    generator)
    # test_dataset = datasets.ImageFolder(CONFIG["test_dir"])
    train_dataset.dataset.targets = list(map(float, train_dataset.dataset.targets))
    validation_dataset.dataset.targets = list(map(float, validation_dataset.dataset.targets))
    test_dataset.dataset.targets = list(map(float, test_dataset.dataset.targets))

    return train_dataset, validation_dataset, test_dataset
