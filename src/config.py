import os

from src import ROOT_DIR

DATA_PATH = os.path.join(ROOT_DIR, "data")
LOGS_PATH = os.path.join(ROOT_DIR, "logs")
TRAIN_CSV = "train_dogs_vs_cats.csv"
TEST_CSV = "test_dogs_vs_cats.csv"

LABEL_MAPPING = {
    "0": "dog",
    "1": "cat"
}
