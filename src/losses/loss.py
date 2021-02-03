import torch

from scipy import stats
from torch.nn.functional import cross_entropy


def pearson_cross_correlation_torch(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return pcc


def pearson_scikit(output, target):
    x = output
    y = target

    pcc, p_value = stats.pearsonr(x, y)
    print("pcc: ", pcc)
    print("p_value: ", p_value)
    return pcc


# Losses
def similarity_loss_pcc(original_explanation: torch.Tensor, adv_explanation: torch.Tensor):
    """
    Calculates the Pearson Cross Correlation between two explanation maps.
    The resulting coefficient is then normed to be in range [0, 1].
    The final loss value is calculated as (1 - pcc_coeff) so as to allow for a loss minimization objective.
    ==> resulting loss: 0 = similar, 1 = dissimilar
    :param original_explanation:
    :param adv_explanation:
    :return:
    """
    pcc_coeff = pearson_cross_correlation_torch(original_explanation, adv_explanation)
    # normalize pcc value to range [0,1]
    pcc_norm = (pcc_coeff + 1.0) / 2.0
    loss = 1.0 - pcc_norm
    return loss


def adv_cross_entropy(model, adv_image, gt_label):
    target_label = 1 - gt_label  # todo: fix dirty hack ;)
    pred_label = model(adv_image)
    adv_loss = cross_entropy(pred_label, target_label)
    return adv_loss


# TODO: add hyperparameters for loss weighing
def combined_loss(model, orig_image, adv_image, orig_explanation, adv_explanation, gt_label):
    pred_label = model(orig_image)
    # Part 1: CrossEntropy for original image
    original_image_loss = cross_entropy(pred_label, gt_label)
    # Part 2: CrossEntropy for adv image
    adv_image_loss = adv_cross_entropy(model, adv_image, gt_label)
    # Part 3: "Similarity" (Pearson Cross Correlation) between original and adversarial explanations
    # returns:
    pcc_loss = similarity_loss_pcc(orig_explanation, adv_explanation)
    loss = original_image_loss + adv_image_loss + pcc_loss
    return loss
