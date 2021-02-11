import matplotlib.pyplot as plt

from torch.nn.functional import cross_entropy

from explanations.captum_explainer import CaptumExplainer
from losses.loss import check_nan
from models.lit_model import LitVGG16Model
from util import rgb_tensor_to_pil_numpy


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
        # set all of networks weights trainable
        if "train_whole_network" in hparams and hparams["train_whole_network"] is True:
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

    # overrides
    def training_step(self, batch, batch_idx):
        original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = batch
        original_explanation_map, adversarial_explanation_map = self._create_explanation_map(original_image,
                                                                                             adversarial_image,
                                                                                             gt_label,
                                                                                             adv_label,
                                                                                             baselines=original_image * 0)
        loss, original_image_loss, adv_image_loss, pcc_loss = self.combined_loss(original_image,
                                                                                 adversarial_image,
                                                                                 original_explanation_map,
                                                                                 adversarial_explanation_map,
                                                                                 gt_label,
                                                                                 adv_label)
        self.log("train_loss_combined", loss, on_step=True, logger=True)
        self.log("train_loss_orig_ce", original_image_loss, on_step=True, logger=True)
        self.log("train_loss_adv_ce", adv_image_loss, on_step=True, logger=True)
        self.log("train_loss_sim", pcc_loss, on_step=True, logger=True)
        if self.global_step % self.image_log_intervals["train"] == 0:
            self.log_explanations_to_tensorboard(original_image,
                                                 original_explanation_map,
                                                 adversarial_image,
                                                 adversarial_explanation_map,
                                                 original_image_name)
        return loss

    def validation_step(self, batch, batch_idx):
        original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = batch
        original_explanation_map, adversarial_explanation_map = self._create_explanation_map(original_image,
                                                                                             adversarial_image,
                                                                                             gt_label,
                                                                                             adv_label,
                                                                                             baselines=original_image * 0)
        loss, original_image_loss, adv_image_loss, pcc_loss = self.combined_loss(original_image,
                                                                                 adversarial_image,
                                                                                 original_explanation_map,
                                                                                 adversarial_explanation_map,
                                                                                 gt_label,
                                                                                 adv_label)
        self.log("val_loss_combined", loss, on_step=True, logger=True)
        self.log("val_loss_orig_ce", original_image_loss, on_step=True, logger=True)
        self.log("val_loss_adv_ce", adv_image_loss, on_step=True, logger=True)
        self.log("val_loss_sim", pcc_loss, on_step=True, logger=True)
        if self.global_step % self.image_log_intervals["val"] == 0:
            self.log_explanations_to_tensorboard(original_image,
                                                 original_explanation_map,
                                                 adversarial_image,
                                                 adversarial_explanation_map,
                                                 original_image_name)
        return loss

    def log_explanations_to_tensorboard(self,
                                        original_image,
                                        original_explanation_map,
                                        adversarial_image,
                                        adversarial_explanation_map,
                                        original_image_name):
        tensorboard = self.logger.experiment
        orig = rgb_tensor_to_pil_numpy(original_image)
        orig_exp = rgb_tensor_to_pil_numpy(original_explanation_map)
        adv = rgb_tensor_to_pil_numpy(adversarial_image)
        adv_exp = rgb_tensor_to_pil_numpy(adversarial_explanation_map)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
        tag_name = f"explanation_comparison_{self.global_step}"
        for i, row_axis in enumerate(axes):
            img_name = original_image_name[i].replace(".jpg", "")
            self.explainer.visualize_single(orig_exp[i],
                                            orig[i],
                                            fig_ax_tuple=(fig, row_axis[0]),
                                            title=f"Original {img_name} ({self.xai_algorithm.__name__})")
            self.explainer.visualize_single(adv_exp[i],
                                            adv[i],
                                            fig_ax_tuple=(fig, row_axis[1]),
                                            title=f"Adversarial {img_name} ({self.xai_algorithm.__name__})")
        tensorboard.add_figure(tag_name, fig, global_step=self.global_step)
        plt.draw()
        plt.show()

    def _create_explanation_map(self, original_image, adversarial_image, gt_label, adv_label, **kwargs):
        original_explanation_maps = self.explainer.explain(original_image,
                                                           gt_label,
                                                           **kwargs)
        adversarial_explanation_maps = self.explainer.explain(adversarial_image,
                                                              adv_label,
                                                              **kwargs)
        return original_explanation_maps, adversarial_explanation_maps
