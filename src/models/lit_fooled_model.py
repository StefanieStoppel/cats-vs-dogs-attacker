from typing import Union, Tuple

import matplotlib.pyplot as plt
import torch

from torch.nn.functional import cross_entropy

from explanations.captum_explainer import CaptumExplainer
from losses.loss import check_nan
from models.lit_model import LitVGG16Model
from util import rgb_tensor_to_pil_numpy
import numpy as np


class LitFooledModel(LitVGG16Model):
    def __init__(self, hparams):
        super(LitFooledModel, self).__init__(hparams)
        self.loss = None
        self.explainer = None

        self.hparams = hparams
        self.similarity_loss = hparams["similarity_loss"]
        self.gammas = hparams["gammas"]
        self.lr = hparams["lr"]
        self.batch_size = hparams["batch_size"]
        self.xai_algorithm = hparams["xai_algorithm"]
        self.image_log_intervals = {
            "train": 50,
            "val": 20
        }

        self.use_fixed_explanations = False
        if "use_fixed_explanations" in hparams and hparams["use_fixed_explanations"] is True:
            self.use_fixed_explanations = True

        self.train_whole_network = False
        # set all of networks weights trainable
        if "train_whole_network" in hparams and hparams["train_whole_network"] is True:
            self.train_whole_network = True
            self.set_requires_grad(True)
        self.save_hyperparameters()

    def set_requires_grad(self, value):
        for param in self.model.parameters():
            param.requires_grad = value

    def set_explainer(self, explainer: CaptumExplainer):
        self.explainer = explainer

    def combined_loss(self, orig_image, adv_image, orig_explanation,
                      adv_explanation, gt_label, adv_label):
        orig_pred_label = self.model(orig_image)

        # Part 1: CrossEntropy for original image
        original_image_loss = cross_entropy(orig_pred_label, gt_label)
        check_nan(original_image_loss, "orig_ce_loss")

        # Part 2: CrossEntropy for adv image
        adv_pred_label = self.model(adv_image)
        adv_image_loss = cross_entropy(adv_pred_label, adv_label)
        check_nan(adv_image_loss, "adv_ce_loss")

        # Part 3: "Similarity" between original and adversarial explanations
        sim_loss = self.similarity_loss(orig_explanation, adv_explanation)
        loss = (self.gammas[0] * original_image_loss) + \
               (self.gammas[1] * adv_image_loss) + \
               (self.gammas[2] * sim_loss)
        return loss, original_image_loss, adv_image_loss, sim_loss

    def training_step(self, batch, batch_idx):
        stage = "train"
        loss = self.predict(batch, stage)
        return loss

    def validation_step(self, batch, batch_idx):
        stage = "val"
        loss = self.predict(batch, stage)
        return loss

    def predict(self, batch, stage):
        if self.use_fixed_explanations:
            original_image, adversarial_image, \
            gt_label, adv_label, original_image_name, adversarial_image_name, original_explanation_map = batch
            original_explanation_map.to(self.device)
        else:
            original_image, adversarial_image, \
            gt_label, adv_label, original_image_name, adversarial_image_name = batch
            original_explanation_map = self.explainer.explain(original_image,
                                                              gt_label)
            # baselines=original_image * 0)

        adversarial_explanation_map = self.explainer.explain(adversarial_image,
                                                             adv_label)
        # baselines=adversarial_image * 0)
        if torch.nonzero(original_explanation_map).numel() == 0:
            print(f"WARNING: explanation for images {original_image_name} contains all zeros!")
        if torch.nonzero(adversarial_explanation_map).numel() == 0:
            print(f"WARNING: explanation for images {adversarial_image_name} contains all zeros!")

        loss, original_image_loss, adv_image_loss, pcc_loss = self.combined_loss(original_image,
                                                                                 adversarial_image,
                                                                                 original_explanation_map,
                                                                                 adversarial_explanation_map,
                                                                                 gt_label,
                                                                                 adv_label)

        self.log(f"{stage}_loss_combined", loss, on_step=True, logger=True)
        self.log(f"{stage}_loss_orig_ce", original_image_loss, on_step=True, logger=True)
        self.log(f"{stage}_loss_adv_ce", adv_image_loss, on_step=True, logger=True)
        self.log(f"{stage}_loss_sim", pcc_loss, on_step=True, logger=True)

        if self.global_step % self.image_log_intervals[stage] == 0:
            self.log_explanations_to_tensorboard(original_image,
                                                 original_explanation_map,
                                                 adversarial_image,
                                                 adversarial_explanation_map,
                                                 original_image_name)
        return loss

    def _create_explanation_map(self, original_image, adversarial_image, gt_label, adv_label, **kwargs):
        original_explanation_maps = self.explainer.explain(original_image,
                                                           gt_label,
                                                           **kwargs)
        adversarial_explanation_maps = self.explainer.explain(adversarial_image,
                                                              adv_label,
                                                              **kwargs)
        return original_explanation_maps, adversarial_explanation_maps

    def log_explanations_to_tensorboard(self,
                                        original_image,
                                        original_explanation_map,
                                        adversarial_image,
                                        adversarial_explanation_map,
                                        original_image_name):
        tensorboard = self.logger.experiment
        tag_name = f"explanation_comparison_{self.global_step}"
        fig = self.create_plt_visualization(original_image,
                                            adversarial_image,
                                            original_explanation_map,
                                            adversarial_explanation_map,
                                            original_image_name)
        tensorboard.add_figure(tag_name, fig, global_step=self.global_step)

    def create_plt_visualization(self,
                                 original_image,
                                 adversarial_image,
                                 original_explanation_map,
                                 adversarial_explanation_map,
                                 original_image_name,
                                 fig_axes_tuple: Union[None, Tuple[plt.figure, plt.axis]] = None,
                                 n_rows=3):
        orig = rgb_tensor_to_pil_numpy(original_image)
        orig_exp = rgb_tensor_to_pil_numpy(original_explanation_map)
        adv = rgb_tensor_to_pil_numpy(adversarial_image)
        adv_exp = rgb_tensor_to_pil_numpy(adversarial_explanation_map)
        sample_indeces = np.arange(self.batch_size)
        # np.random.shuffle(sample_indeces, )
        if fig_axes_tuple is None:
            fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 12))
        else:
            fig, axes = fig_axes_tuple
        for i, row_axis in enumerate(axes):
            batch_idx = sample_indeces[i]
            if np.count_nonzero(orig_exp[batch_idx]) == 0:
                print(f"WARNING: cannot visualize - explanation for image "
                      f"{original_image_name[batch_idx]} contains all zeros!")
                continue
            if np.count_nonzero(adv_exp[batch_idx]) == 0:
                print(f"WARNING: cannot visualize -  explanation for adversarial image "
                      f"{original_image_name[batch_idx]} contains all zeros!")
                continue

            img_name = original_image_name[batch_idx].replace(".jpg", "")
            self.explainer.visualize_single(orig_exp[batch_idx],
                                            orig[batch_idx],
                                            fig_ax_tuple=(fig, row_axis[0]),
                                            title=f"Original {img_name} ({self.xai_algorithm.__name__})")
            self.explainer.visualize_single(adv_exp[batch_idx],
                                            adv[batch_idx],
                                            fig_ax_tuple=(fig, row_axis[1]),
                                            title=f"Adversarial {img_name} ({self.xai_algorithm.__name__})")
        return fig
