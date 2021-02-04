import os
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from util import pil_read


class DogVsCatWithAdversarialsDataset(Dataset):
    def __init__(self, data_txt_file, dirname, mode='train', transform=None):
        self.data = pd.read_csv(data_txt_file, delimiter=",", header=None, names=["orig", "orig_idx", "adv", "adv_idx"])
        self.data = self.data.to_numpy()
        self.orig_idx, self.orig_label_idx, self.adv_idx, self.adv_label_idx = 0, 1, 2, 3
        self.dirname = dirname
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        orig_img_name = row[self.orig_idx]
        orig_img = pil_read(os.path.join(self.dirname, orig_img_name))
        orig_label = row[self.orig_label_idx]

        adv_img_name = row[self.adv_idx]
        adv_img = pil_read(os.path.join(self.dirname, adv_img_name))
        adv_label = row[self.adv_label_idx]

        if self.transform:
            orig_img = self.transform(orig_img)
            adv_img = self.transform(adv_img)
        return orig_img, adv_img, orig_label, adv_label, orig_img_name, adv_img_name


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


if __name__ == '__main__':
    csv_path = "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/data_adv/train_adv.csv"
    dirname = "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/data_adv/train"
    dataset = DogVsCatWithAdversarialsDataset(csv_path, dirname)
    dataset.__getitem__(2493)