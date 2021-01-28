import os
import pandas as pd

from typing import List

from src.config import DATA_PATH, TRAIN_CSV, TEST_CSV


def save_dogs_vs_cats_dataframes(data_path, target_paths: List[str]):
    image_col = "image"
    label_col = "label"
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")

    train_images, train_labels = get_images_and_labels(train_dir)
    train_dogs_vs_cats = pd.DataFrame({image_col: train_images, label_col: train_labels})

    test_images, _ = get_images_and_labels(test_dir, is_test=True)
    test_dogs_vs_cats = pd.DataFrame({image_col: test_images})

    train_dogs_vs_cats.to_csv(os.path.abspath(target_paths[0]), index=False)
    test_dogs_vs_cats.to_csv(os.path.abspath(target_paths[1]), index=False)


def get_images_and_labels(dir_path, is_test=False):
    images, labels = list(), list()
    for file in os.listdir(dir_path):
        label = "0"
        if file.startswith("cat"):
            label = "1"
        images.append(file)
        labels.append(label)
    if is_test:
        return images, None
    return images, labels


def load_dogs_vs_cats_dataframes(train_csv, test_csv):
    return pd.read_csv(train_csv), pd.read_csv(test_csv)


if __name__ == '__main__':
    train_csv_path = os.path.join(DATA_PATH, TRAIN_CSV)
    test_csv_path = os.path.join(DATA_PATH, TEST_CSV)
    save_dogs_vs_cats_dataframes(DATA_PATH, [train_csv_path, test_csv_path])
