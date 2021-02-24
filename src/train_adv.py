import csv
import os

import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from captum.attr import DeepLift, LayerGradCam
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from operator import itemgetter
from pytorch_lightning import loggers as pl_loggers, Trainer
from torchvision import transforms
from tqdm import tqdm

from config import DATA_ADV_PATH, LOGS_PATH, LOGS_ADV_PATH
from dataloader import DogVsCatWithAdversarialsDataset, DogVsCatWithOriginalExplanationsDataset
from models.lit_fooled_model import LitFooledModel
from util import map_explanations_forward
from explanations.captum_explainer import get_explainer


def run_train_adv(config):
    train_dir, train_csv, validation_dir, validation_csv, \
    logs_dir, num_workers, data_transforms, hparams, hparams_file, checkpoint = itemgetter("train_dir",
                                                                                           "train_csv",
                                                                                           "validation_dir",
                                                                                           "validation_csv",
                                                                                           "logs_dir",
                                                                                           "num_workers",
                                                                                           "transform",
                                                                                           "hparams",
                                                                                           "hparams_file",
                                                                                           "checkpoint")(config)
    batch_size = hparams["batch_size"]
    xai_algorithm = hparams["xai_algorithm"]
    xai_algorithm_kwargs = hparams["xai_algorithm_kwargs"]
    use_fixed_explanations = hparams["use_fixed_explanations"]

    # lit_fooled_model = LitFooledModel(hparams)
    lit_fooled_model = LitFooledModel(hparams).load_from_checkpoint(
        checkpoint_path=checkpoint,
        hparams=hparams
    )

    # Explainer
    explainer = get_explainer(xai_algorithm, lit_fooled_model, **xai_algorithm_kwargs)
    lit_fooled_model.set_explainer(explainer)

    if use_fixed_explanations:
        train_dataset = DogVsCatWithOriginalExplanationsDataset(train_csv,
                                                                train_dir,
                                                                transform=data_transforms,
                                                                xai_algorithm_name=xai_algorithm.__name__)
        validation_dataset = DogVsCatWithOriginalExplanationsDataset(validation_csv,
                                                                     validation_dir,
                                                                     transform=data_transforms,
                                                                     xai_algorithm_name=xai_algorithm.__name__)
    else:
        train_dataset = DogVsCatWithAdversarialsDataset(train_csv,
                                                        train_dir,
                                                        transform=data_transforms)
        validation_dataset = DogVsCatWithAdversarialsDataset(validation_csv,
                                                             validation_dir,
                                                             transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

    # init logging
    tb_logger = pl_loggers.TensorBoardLogger(logs_dir)
    # trainer = Trainer(logger=tb_logger, gpus=1, resume_from_checkpoint=checkpoint2)
    trainer = Trainer(logger=tb_logger, gpus=1)

    # run training
    trainer.fit(lit_fooled_model, train_loader, validation_loader)


def run_test_sim(config):
    test_dir, test_csv, num_workers, data_transforms, checkpoint, checkpoint2, hparams, hparams_file = \
        itemgetter(
            "test_dir",
            "test_csv",
            "num_workers",
            "test_transform",
            "checkpoint",
            "checkpoint_2",
            "hparams",
            "hparams_file")(config)

    batch_size = hparams["batch_size"]
    xai_algorithm = hparams["xai_algorithm"]
    xai_algorithm_kwargs = hparams["xai_algorithm_kwargs"]
    similarity_loss = hparams["similarity_loss"]
    use_fixed_explanations = hparams["use_fixed_explanations"]

    if use_fixed_explanations:
        test_dataset = DogVsCatWithOriginalExplanationsDataset(test_csv,
                                                               test_dir,
                                                               transform=data_transforms,
                                                               xai_algorithm_name=xai_algorithm.__name__)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    else:
        test_dataset = DogVsCatWithAdversarialsDataset(test_csv,
                                                       test_dir,
                                                       transform=data_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    #### 1) first model explanations
    lit_fooled_model = LitFooledModel(hparams).load_from_checkpoint(
        checkpoint_path=checkpoint,
        hparams_file=hparams_file
    )
    lit_fooled_model.cuda()
    explainer = get_explainer(xai_algorithm, lit_fooled_model, **xai_algorithm_kwargs)
    lit_fooled_model.set_explainer(explainer)

    explanation_maps = {
        "original_gt": [],
        "pre_training": {
            "original": [],
            "adversarial": [],
            "loss": {
                "sim": [],
                "total": [],
                "orig_ce": [],
                "adv_ce": []
            }
        },
        "post_training": {
            "original": [],
            "adversarial": [],
            "loss": {
                "sim": [],
                "total": [],
                "orig_ce": [],
                "adv_ce": []
            }
        }
    }
    print("Iterate test data with model pre-training")
    for i, batch in enumerate(tqdm(test_loader)):
        original_image, adversarial_image, gt_label, adv_label, \
        original_image_name, adversarial_image_name, orig_explanation_map_gt = batch

        first_original_explanation_maps, first_adversarial_explanation_maps = \
            lit_fooled_model._create_explanation_map(original_image.cuda(),
                                                     adversarial_image.cuda(),
                                                     gt_label.cuda(),
                                                     adv_label.cuda())
        # explanation_maps["original_gt"].append(orig_explanation_map_gt)
        # explanation_maps["pre_training"]["original"].append(first_original_explanation_maps)
        # explanation_maps["pre_training"]["adversarial"].append(first_adversarial_explanation_maps)

        loss, original_image_loss, adv_image_loss, sim_loss \
            = lit_fooled_model.combined_loss(original_image.cuda(),
                                             adversarial_image.cuda(),
                                             first_original_explanation_maps.cuda(),
                                             first_adversarial_explanation_maps.cuda(),
                                             gt_label.cuda(),
                                             adv_label.cuda())
        explanation_maps["pre_training"]["loss"]["sim"].append(sim_loss.item())
        explanation_maps["pre_training"]["loss"]["total"].append(loss.item())
        explanation_maps["pre_training"]["loss"]["orig_ce"].append(original_image_loss.item())
        explanation_maps["pre_training"]["loss"]["adv_ce"].append(adv_image_loss.item())

        if i % 100 == 0:
            # plt.ion()
            fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(12, 12))
            fig = lit_fooled_model.create_plt_visualization(original_image,
                                                            adversarial_image,
                                                            first_original_explanation_maps,
                                                            first_adversarial_explanation_maps,
                                                            original_image_name,
                                                            (fig, axes),
                                                            n_rows=batch_size)
            fig.savefig(f"../plots/explanation_maps/run_180/pre-training_test-batch-{i}.jpg")


    # 2) second model explanations
    lit_fooled_model = LitFooledModel(hparams).load_from_checkpoint(
        checkpoint_path=checkpoint2,
        hparams_file=hparams_file
    )
    lit_fooled_model.cuda()

    explainer = get_explainer(xai_algorithm, lit_fooled_model, **xai_algorithm_kwargs)
    lit_fooled_model.set_explainer(explainer)

    print("Iterate test data with model post-training")
    for i, batch in enumerate(tqdm(test_loader)):
        original_image, adversarial_image, gt_label, adv_label, \
        original_image_name, adversarial_image_name, orig_explanation_map_gt = batch

        second_original_explanation_maps, second_adversarial_explanation_maps = \
            lit_fooled_model._create_explanation_map(original_image.cuda(),
                                                     adversarial_image.cuda(),
                                                     gt_label.cuda(),
                                                     adv_label.cuda())
        # explanation_maps["post_training"]["original"].append(second_original_explanation_maps)
        # explanation_maps["post_training"]["adversarial"].append(second_adversarial_explanation_maps)

        loss, original_image_loss, adv_image_loss, sim_loss \
            = lit_fooled_model.combined_loss(original_image.cuda(),
                                             adversarial_image.cuda(),
                                             second_original_explanation_maps.cuda(),
                                             second_adversarial_explanation_maps.cuda(),
                                             gt_label.cuda(),
                                             adv_label.cuda())
        explanation_maps["post_training"]["loss"]["sim"].append(sim_loss.item())
        explanation_maps["post_training"]["loss"]["total"].append(loss.item())
        explanation_maps["post_training"]["loss"]["orig_ce"].append(original_image_loss.item())
        explanation_maps["post_training"]["loss"]["adv_ce"].append(adv_image_loss.item())

        if i % 100 == 0:
            fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(12, 12))
            fig = lit_fooled_model.create_plt_visualization(original_image,
                                                            adversarial_image,
                                                            second_original_explanation_maps,
                                                            second_adversarial_explanation_maps,
                                                            original_image_name,
                                                            (fig, axes),
                                                            n_rows=batch_size)
            fig.savefig(f"../plots/explanation_maps/run_180/post-training_test-batch-{i}.jpg")

    # calculate similarity losses
    pre_training_loss_total = np.mean(explanation_maps["pre_training"]["loss"]["total"])
    post_training_loss_total = np.mean(explanation_maps["post_training"]["loss"]["total"])
    pre_training_loss_sim = np.mean(explanation_maps["pre_training"]["loss"]["sim"])
    post_training_loss_sim = np.mean(explanation_maps["post_training"]["loss"]["sim"])
    pre_training_loss_orig_ce = np.mean(explanation_maps["pre_training"]["loss"]["orig_ce"])
    post_training_loss_orig_ce = np.mean(explanation_maps["post_training"]["loss"]["orig_ce"])
    pre_training_loss_adv_ce = np.mean(explanation_maps["pre_training"]["loss"]["adv_ce"])
    post_training_loss_adv_ce = np.mean(explanation_maps["post_training"]["loss"]["adv_ce"])

    print(f"Used similarity loss: {similarity_loss.__name__}")
    print(f"*Pre-training* total loss for orig and adv explanation maps: {pre_training_loss_total}")
    print(f"*Post-training* total loss for orig and adv explanation maps: {post_training_loss_total}")

    print(f"*Pre-training* similarity loss for orig and adv explanation maps: {pre_training_loss_sim}")
    print(f"*Post-training* similarity loss for orig and adv explanation maps: {post_training_loss_sim}")

    print(f"*Pre-training* orig CE loss: {pre_training_loss_orig_ce}")
    print(f"*Post-training* orig CE loss: {post_training_loss_orig_ce}")

    print(f"*Pre-training* adv CE loss: {pre_training_loss_adv_ce}")
    print(f"*Post-training* adv CE loss: {post_training_loss_adv_ce}")


def prepare_data_with_predictions(config):
    train_dir, train_csv, validation_dir, validation_csv, test_dir, test_csv, \
    logs_dir, num_workers, data_transforms, checkpoint, hparams = itemgetter("train_dir",
                                                                             "train_csv",
                                                                             "validation_dir",
                                                                             "validation_csv",
                                                                             "test_dir",
                                                                             "test_csv",
                                                                             "logs_dir",
                                                                             "num_workers",
                                                                             "transform",
                                                                             "checkpoint",
                                                                             "hparams")(config)
    batch_size = hparams["batch_size"]
    xai_algorithm = hparams["xai_algorithm"]
    xai_algorithm_kwargs = hparams["xai_algorithm_kwargs"]

    lit_fooled_model = LitFooledModel.load_from_checkpoint(
        checkpoint_path=checkpoint,
        hparams_file="/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/logs_adv/default/version_79/hparams.yaml"
    )

    #  XAI algorithm
    explainer = get_explainer(xai_algorithm, lit_fooled_model, **xai_algorithm_kwargs)
    lit_fooled_model.set_explainer(explainer)
    lit_fooled_model.cuda()
    lit_fooled_model.eval()

    # Dataset
    train_dataset = DogVsCatWithAdversarialsDataset(train_csv,
                                                    train_dir,
                                                    transform=data_transforms)
    validation_dataset = DogVsCatWithAdversarialsDataset(validation_csv,
                                                         validation_dir,
                                                         transform=data_transforms)
    test_dataset = DogVsCatWithAdversarialsDataset(test_csv,
                                                   test_dir,
                                                   transform=data_transforms)
    #  DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    save_explanations(explainer, train_loader, train_dir, xai_algorithm.__name__, save_as_jpg=False)
    save_explanations(explainer, validation_loader, validation_dir, xai_algorithm.__name__, save_as_jpg=False)
    save_explanations(explainer, test_loader, test_dir, xai_algorithm.__name__, save_as_jpg=False)


def save_explanations(explainer, data_loader, data_dir, xai_algorithm_name, save_as_jpg=True):
    all_zeros = list()
    for batch in tqdm(data_loader):
        orig_img, _, orig_label, _, orig_img_name, _ = batch
        orig_img = orig_img.cuda()
        orig_label = orig_label.cuda()
        explanations = explainer.explain(orig_img, orig_label)
        explanations_mapped = map_explanations_forward(explanations)
        # explanations_b = map_explanations_backward(explanations_mapped)

        for orig_name, explanation_mapped, explanation in zip(orig_img_name, explanations_mapped, explanations):
            if torch.nonzero(explanation).numel() == 0:
                all_zeros.append(os.path.join(data_dir, orig_name))
                print(f"WARNING: explanation for image {orig_name} contains all zeros!")
            if save_as_jpg:
                explanation_path = os.path.join(data_dir,
                                                orig_name.replace("orig.jpg", f"exp_{xai_algorithm_name}.jpg"))
                torchvision.utils.save_image(explanation_mapped, explanation_path)
            else:
                explanation_path = os.path.join(data_dir,
                                                orig_name.replace("orig.jpg", f"exp_{xai_algorithm_name}.pt"))
                # save as tensor when image has negative values
                torch.save(explanation, explanation_path)

    print(f"number of all zeros explanations: {len(all_zeros)}")
    with open("exp_all_zeros.csv", 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(all_zeros)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')  # to be able to use torch.load in dataloader

    CONFIG = {
        # Directories
        "train_dir": os.path.join(DATA_ADV_PATH, "train"),
        "validation_dir": os.path.join(DATA_ADV_PATH, "validation"),
        "test_dir": os.path.join(DATA_ADV_PATH, "test"),

        "train_csv": os.path.join(DATA_ADV_PATH, "train_adv.csv"),
        "validation_csv": os.path.join(DATA_ADV_PATH, "validation_adv.csv"),
        "test_csv": os.path.join(DATA_ADV_PATH, "test_adv.csv"),

        "logs_dir": LOGS_ADV_PATH,
        # checkpoint of unmodified cat-dog classifier
        "checkpoint": os.path.join(LOGS_PATH, "default/version_13/checkpoints/epoch=0-step=136.ckpt"),
        # checkpoint of adversarially trained cat-dog classifier
        "checkpoint_2": os.path.join(LOGS_ADV_PATH, "default/version_180/checkpoints/epoch=1-step=2843.ckpt"),
        "hparams_file": os.path.join(LOGS_ADV_PATH, "default/version_180/hparams.yaml"),

        # Transform images
        # "transform": transforms.Compose([transforms.RandomRotation(30),
        #                                  transforms.RandomResizedCrop(224),
        #                                  transforms.RandomHorizontalFlip(),
        #                                  transforms.ToTensor(),
        #                                  transforms.Normalize([0.485, 0.456, 0.406],
        #                                                       [0.229, 0.224, 0.225])]),
        "transform": transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
        "test_transform": transforms.Compose([transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])]),

        # Training
        "hparams": {
            "lr": 5e-5,
            "batch_size": 8,
            "num_classes": 2,
            "gammas": (1, 1, 1e6),
            "xai_algorithm": LayerGradCam,
            "xai_algorithm_kwargs": dict(
                relu_attributions=False
            ),
            "similarity_loss": mse_loss,
            "train_whole_network": True,
            "use_fixed_explanations": True
        },
        "num_workers": 2,
    }
    # run_train_adv(config=CONFIG)
    run_test_sim(config=CONFIG)
    # prepare_data_with_predictions(config=CONFIG)
