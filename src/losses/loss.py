import torch

from pytorch_msssim import ssim
from scipy import stats

from util import map_image_to_unit_interval


def pearson_cross_correlation_torch(output, target):
    x = output
    y = target

    x_n = map_image_to_unit_interval(x)
    y_n = map_image_to_unit_interval(y)

    vx = x_n - torch.mean(x_n)
    vy = y_n - torch.mean(y_n)

    double_sum_vx = torch.sum(vx ** 2)
    double_sum_vy = torch.sum(vy ** 2)
    if double_sum_vy < 0 or double_sum_vy < 0:
        raise SystemExit(f"Sum was < 0: PCC cannot be calculated for negative numbers due to sqrt().")

    pcc = torch.sum(vx * vy) / (torch.sqrt(double_sum_vx) * torch.sqrt(double_sum_vy))
    if torch.isnan(pcc).any():
        raise SystemExit(f"NaN in pcc!")
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
    if torch.isnan(loss).any():
        raise SystemExit(f"NaN in pcc loss!")
    return loss


def similarity_loss_ssim(original_explanation: torch.Tensor, adv_explanation: torch.Tensor, size_average=True):
    ssim_ = ssim(original_explanation, adv_explanation, data_range=1.0, size_average=size_average)
    ssim_loss = 1 - ssim_
    return ssim_loss


def check_nan(tensor, loss_name):
    if torch.isnan(tensor).any():
        raise SystemExit(f"NaN in {loss_name}!")
    return
