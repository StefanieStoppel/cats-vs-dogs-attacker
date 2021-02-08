import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchvision import models

from src.util import weight_reset


class LitVGG16Model(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.lr = hparams["lr"]
        self.num_classes = hparams["num_classes"]
        self.save_hyperparameters()

        self.loss = CrossEntropyLoss()

        # metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.validation_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        self.model = models.vgg16(pretrained=True)

        # freeze layers
        for param in self.model.parameters():
            param.requires_grad = False

        # add layers to classification head, reset weights and make it trainable
        self.model.classifier = self._customize_classification_head(self.model.classifier, self.num_classes)

    @staticmethod
    def _customize_classification_head(classification_head, num_classes):
        classification_head_ = classification_head
        classification_head_.apply(weight_reset)
        classification_head_.add_module("7", torch.nn.Linear(in_features=1000, out_features=num_classes))

        # set classifier layers trainable
        for layer in classification_head_:
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                layer.weight.requires_grad = True
                layer.bias.requires_grad = True

        return classification_head_

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _predict(self, batch):
        image, label, _ = batch
        pred_label = self.model(image)
        loss = self.loss(pred_label, label)
        return label, loss, pred_label

    def training_step(self, batch, batch_idx):
        label, loss, pred_label = self._predict(batch)
        self.log("train_loss", loss, on_step=True, logger=True)
        self.train_accuracy(pred_label, label)
        return loss

    def validation_step(self, batch, batch_idx):
        label, loss, pred_label = self._predict(batch)
        self.log("val_loss", loss, logger=True)
        self.validation_accuracy(pred_label, label)
        return loss

    def test_step(self, batch, batch_idx):
        label, loss, pred_label = self._predict(batch)
        self.log("test_loss", loss, logger=True)
        self.test_accuracy(pred_label, label)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute())

    def test_epoch_end(self, outs):
        self.log("test_acc_epoch", self.test_accuracy.compute())

    @staticmethod
    def _print_correct_ratio(label, pred_label):
        predicted_labels = pred_label.argmax(-1)
        batch_size = len(label)
        correct_count = torch.sum((predicted_labels == label)).item()
        print("\ncorrect / batch: ", correct_count, "/", batch_size)


if __name__ == '__main__':
    lit = LitVGG16Model()
