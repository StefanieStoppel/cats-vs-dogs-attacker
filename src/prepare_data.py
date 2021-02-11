import os
import pandas as pd

from typing import List

from src import ROOT_DIR


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


def save_dogs_vs_cats_file_list(data_path, target_file_name):
    file_list = list()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jpg"):
                parent_dir = os.path.basename(root)
                file_list.append(f"{parent_dir}/{file}")
    pd.DataFrame(file_list).to_csv(os.path.join(data_path, target_file_name), index=False, header=False)


def save_dogs_vs_cats_adversarial_file_list(data_path, target_file_path):
    file_list = list()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("orig.jpg"):
                label = 0
                adv_label = 1
                if "dog" in file:
                    label = 1
                    adv_label = 0
                adv_file = file.replace("orig.jpg", "adv.jpg")
                file_list.append((file, label, adv_file, adv_label))
    pd.DataFrame(file_list).to_csv(os.path.join(target_file_path), index=False, header=False)


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
    # train_csv_path = os.path.join(DATA_PATH, TRAIN_CSV)
    # test_csv_path = os.path.join(DATA_PATH, TEST_CSV)
    # save_dogs_vs_cats_dataframes(DATA_PATH, [train_csv_path, test_csv_path])
    # save_dogs_vs_cats_file_list(os.path.join(DATA_PATH, "train"), os.path.join(DATA_PATH, "train_list.csv"))
    # save_dogs_vs_cats_file_list(os.path.join(DATA_PATH, "test"), os.path.join(DATA_PATH, "test_list.csv"))
    # save_dogs_vs_cats_adversarial_file_list(os.path.join(ROOT_DIR, "data/data_adv/train"),
    #                                         os.path.join(ROOT_DIR, "data/data_adv/train_adv.csv"))
    save_dogs_vs_cats_adversarial_file_list(os.path.join(ROOT_DIR, "data/data_adv/validation"),
                                            os.path.join(ROOT_DIR, "data/data_adv/validation_adv.csv"))
    save_dogs_vs_cats_adversarial_file_list(os.path.join(ROOT_DIR, "data/data_adv/test"),
                                            os.path.join(ROOT_DIR, "data/data_adv/test_adv.csv"))
