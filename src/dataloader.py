import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets


def get_dogs_vs_cats_data_splits(data_dir, data_split=(0.7, 0.2, 0.1), transform=None, random_seed=42):
    ImageFolder_ = dataset_with_file_names(datasets.ImageFolder)
    dogs_vs_cats_dataset = ImageFolder_(data_dir, transform=transform)

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
    train_dataset.dataset.targets = list(map(float, train_dataset.dataset.targets))
    validation_dataset.dataset.targets = list(map(float, validation_dataset.dataset.targets))
    test_dataset.dataset.targets = list(map(float, test_dataset.dataset.targets))

    return train_dataset, validation_dataset, test_dataset


def dataset_with_file_names(cls):
    """
    Modifies the given PyTorch Dataset class to return a tuple data, target, index
    instead of just data, target.

    Source: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        file_name = os.path.basename(self.imgs[index][0])
        return data, target, file_name

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
