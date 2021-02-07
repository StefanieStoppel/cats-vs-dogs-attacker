import os
import torch
from captum.attr import DeepLift
from torchvision.transforms import transforms

from config import LOGS_PATH
from explanations.captum_explainer import CaptumExplainer
from models.lit_model import LitVGG16Model
from losses.loss import combined_loss, similarity_loss_ssim
from util import load_image_as_numpy_array, pil_read


if __name__ == '__main__':
    CONFIG = {
        # Paths
        "original_image_path": "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/adversarials/LinfFastGradientAttack/0.005/cat.11489_orig.jpg",
        "adversarial_image_path": "/home/steffi/dev/master_thesis/cats-vs-dogs-attacker/data/adversarials/LinfFastGradientAttack/0.005/cat.11489_adv.jpg",
        "checkpoint": os.path.join(LOGS_PATH, "default/version_10/checkpoints/epoch=0-step=136.ckpt"),

        # other
        "random_seed": 42,
        "transform": transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])]),
        "use_cuda": True
    }

    # GPU or CPU
    device = torch.device('cuda' if (torch.cuda.is_available() and CONFIG["use_cuda"]) else 'cpu')

    # Load model
    lit_model = LitVGG16Model.load_from_checkpoint(checkpoint_path=CONFIG["checkpoint"])
    # model = torchvision.models.vgg16(pretrained=True)
    model = lit_model.model
    model = model.to(device)

    # Explainer
    deeplift_explainer = CaptumExplainer(DeepLift, model)

    # Load images as numpy arrays
    original_img_np = load_image_as_numpy_array(CONFIG["original_image_path"])
    adversarial_img_np = load_image_as_numpy_array(CONFIG["adversarial_image_path"])

    # Load images as PyTorch Tensors
    transform = CONFIG["transform"]
    images_tensor = torch.stack(((transform(pil_read(CONFIG["original_image_path"]))),
                                 (transform(pil_read(CONFIG["adversarial_image_path"])))),
                                dim=0
                                ).to(device)
    # labels_tensor = torch.tensor((990, 73)).to(device)

    # cat: 0; dog: 1
    labels_tensor = torch.tensor((0, 1)).to(device)
    plot_titles = ("DeepLIFT for cat (0, original)", "DeepLIFT for dog (1, adversarial)")

    # Create explanations
    attributions = deeplift_explainer.explain(images_tensor, labels_tensor, baselines=images_tensor * 0)

    ssim_loss = similarity_loss_ssim(images_tensor[0].unsqueeze(0), images_tensor[1].unsqueeze(0))

    loss = combined_loss(model,
                         images_tensor[0].unsqueeze(0),
                         images_tensor[1].unsqueeze(0),
                         attributions[0],
                         attributions[1],
                         labels_tensor[0].unsqueeze(0),
                         labels_tensor[1].unsqueeze(0))
    print("total_loss: ", loss)

    # Visualize
    deeplift_explainer.visualize(attributions, images_tensor, titles=plot_titles)
