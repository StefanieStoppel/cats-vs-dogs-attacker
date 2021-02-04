from explanations.captum_explainer import CaptumExplainer
from losses.loss import combined_loss
from models.lit_model import LitVGG16Model


class LitFooledModel(LitVGG16Model):
    def __init__(self, lr=1e-3, num_classes=2):
        super(LitFooledModel, self).__init__(lr=lr, num_classes=num_classes)
        self.loss = None
        self.combined_loss = combined_loss
        self.explainer = None

    def set_explainer(self, explainer: CaptumExplainer):
        self.explainer = explainer

    # overrides
    def training_step(self, batch, batch_idx):
        original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = batch
        original_explanation_map, adversarial_explanation_map = self._create_explanation_map(original_image,
                                                                                             adversarial_image,
                                                                                             gt_label,
                                                                                             adv_label)
        loss, original_image_loss, adv_image_loss, pcc_loss = self.combined_loss(self.model,
                                                                                 original_image,
                                                                                 adversarial_image,
                                                                                 original_explanation_map,
                                                                                 adversarial_explanation_map,
                                                                                 gt_label,
                                                                                 adv_label)
        self.log("train_loss_combined", loss, on_step=True, logger=True)
        self.log("train_loss_orig_ce", original_image_loss, on_step=True, logger=True)
        self.log("train_loss_adv_ce", adv_image_loss, on_step=True, logger=True)
        self.log("train_loss_sim_pcc", pcc_loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = batch
        original_explanation_map, adversarial_explanation_map = self._create_explanation_map(original_image,
                                                                                             adversarial_image,
                                                                                             gt_label,
                                                                                             adv_label)
        loss, original_image_loss, adv_image_loss, pcc_loss = self.combined_loss(self.model,
                                                                                 original_image,
                                                                                 adversarial_image,
                                                                                 original_explanation_map,
                                                                                 adversarial_explanation_map,
                                                                                 gt_label,
                                                                                 adv_label)
        self.log("val_loss_combined", loss, on_step=True, logger=True)
        self.log("val_loss_orig_ce", original_image_loss, on_step=True, logger=True)
        self.log("val_loss_adv_ce", adv_image_loss, on_step=True, logger=True)
        self.log("val_loss_sim_pcc", pcc_loss, on_step=True, logger=True)
        return loss

    def _create_explanation_map(self, original_image, adversarial_image, gt_label, adv_label):
        # adv_label = 1 - gt_label
        original_explanation_maps = self.explainer.explain(self.model,
                                                           original_image,
                                                           gt_label,
                                                           baselines=original_image * 0)
        adversarial_explanation_maps = self.explainer.explain(self.model,
                                                              adversarial_image,
                                                              adv_label,
                                                              baselines=adversarial_image * 0)
        return original_explanation_maps, adversarial_explanation_maps
