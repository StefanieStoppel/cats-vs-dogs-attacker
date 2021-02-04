import os

from src import ROOT_DIR

DATA_PATH = os.path.join(ROOT_DIR, "data")
DATA_ADV_PATH = os.path.join(DATA_PATH, "data_adv")
LOGS_PATH = os.path.join(ROOT_DIR, "logs")
LOGS_ADV_PATH = os.path.join(ROOT_DIR, "logs_adv")
TRAIN_CSV = "train_dogs_vs_cats.csv"
TEST_CSV = "test_dogs_vs_cats.csv"

LABEL_MAPPING = {
    0: "cat",
    1: "dog"
}
