import torch
import pytorch_lightning as pl
from torch.nn import BCEWithLogitsLoss
from torchvision import models

from src.util import weight_reset


class LitVGG16Model(pl.LightningModule):

    def __init__(self, lr=1e-3, num_classes=1):
        super().__init__()
        self.lr = lr
        self.loss = BCEWithLogitsLoss()
        self.model = models.vgg16(pretrained=True)
        # freeze layers
        for param in self.model.parameters():
            param.requires_grad = False

        # reset and change classification head
        classification_head = self.model.classifier
        classification_head.apply(weight_reset)
        classification_head.add_module("7", torch.nn.Linear(in_features=1000, out_features=num_classes))
        self.model.classifier = classification_head

        # set classifier layers trainable
        for layer in self.model.classifier:
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, label = batch
        pred_label = self.model(image)
        pred_label = pred_label.flatten()
        label = label.type_as(pred_label)
        loss = self.loss(pred_label, label)
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred_label = self.model(image)
        pred_label = pred_label.flatten()
        label = label.type_as(pred_label)
        val_loss = self.loss(pred_label, label)
        self.log("val_loss", val_loss, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        image, label = batch
        pred_label = self.model(image)
        pred_label = pred_label.flatten()
        label = label.type_as(pred_label)
        loss = self.loss(pred_label, label)
        self.log("test_loss", loss, logger=True)
        return loss


if __name__ == '__main__':
    lit = LitVGG16Model()
    print()
