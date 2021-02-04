import torch

from scipy import stats
from torch.nn.functional import cross_entropy


def pearson_cross_correlation_torch(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    # if torch.count_nonzero(vx) == 0:
    #     raise SystemExit(f"First tensor of PCC was all zeros. Cannot calculate correlation coefficient.")

    double_sum_vx = torch.sum(vx ** 2)
    double_sum_vy = torch.sum(vy ** 2)

    if double_sum_vy < 0 or double_sum_vy < 0:
        raise SystemExit(f"Sum was < 0: PCC cannot be calculated for negative numbers due to sqrt().")

    pcc = torch.sum(vx * vy) / (torch.sqrt(double_sum_vx) * torch.sqrt(double_sum_vy))
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


def adv_cross_entropy(adv_gt_label, adv_pred_label):
    adv_loss = cross_entropy(adv_pred_label, adv_gt_label)
    return adv_loss


def combined_loss(model, orig_image, adv_image, orig_explanation,
                  adv_explanation, gt_label, adv_label, gammas=(1, 1, 2)):
    orig_pred_label = model(orig_image)
    # Part 1: CrossEntropy for original image
    original_image_loss = cross_entropy(orig_pred_label, gt_label)
    # Part 2: CrossEntropy for adv image
    adv_pred_label = model(adv_image)
    adv_image_loss = adv_cross_entropy(adv_label, adv_pred_label)
    # Part 3: "Similarity" (Pearson Cross Correlation) between original and adversarial explanations
    # returns:
    pcc_loss = similarity_loss_pcc(orig_explanation, adv_explanation)
    loss = (gammas[0] * original_image_loss) + (gammas[1] * adv_image_loss) + (gammas[2] * pcc_loss)
    return loss, original_image_loss, adv_image_loss, pcc_loss
