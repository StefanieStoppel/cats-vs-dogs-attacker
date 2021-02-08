from torch.nn.functional import cross_entropy

from explanations.captum_explainer import CaptumExplainer
from losses.loss import check_nan
from models.lit_model import LitVGG16Model


class LitFooledModel(LitVGG16Model):
    def __init__(self, hparams):
        super(LitFooledModel, self).__init__(lr=hparams["lr"], num_classes=2)
        self.loss = None
        self.explainer = None

        self.hparams = hparams
        self.similarity_loss = hparams["similarity_loss"]
        self.gammas = hparams["gammas"]
        self.lr = hparams["lr"]

        self.save_hyperparameters()

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
                                                                                             adv_label)
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
        return loss

    def validation_step(self, batch, batch_idx):
        original_image, adversarial_image, gt_label, adv_label, original_image_name, adversarial_image_name = batch
        original_explanation_map, adversarial_explanation_map = self._create_explanation_map(original_image,
                                                                                             adversarial_image,
                                                                                             gt_label,
                                                                                             adv_label)
        loss, original_image_loss, adv_image_loss, pcc_loss = self.combined_loss(original_image,
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
        original_explanation_maps = self.explainer.explain(original_image,
                                                           gt_label,
                                                           baselines=original_image * 0)
        adversarial_explanation_maps = self.explainer.explain(adversarial_image,
                                                              adv_label,
                                                              baselines=adversarial_image * 0)
        return original_explanation_maps, adversarial_explanation_maps
