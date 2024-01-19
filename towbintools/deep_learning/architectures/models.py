import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score
from towbintools.deep_learning.utils.loss import FocalTverskyLoss
import pretrained_microscopy_models as pmm
from .archs import Unet, UnetPlusPlus
import torch


class PretrainedSegmentationModel(pl.LightningModule):
    """Pytorch Lightning Module for training a segmentation model with a pretrained encoder. The encoder is loaded from the pretrained_microscopy_models package.
    Because of the way they were pretrained, input images are required to have 3 channels.

    Parameters:
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            architecture (str): The architecture of the segmentation model.
            encoder (str): The encoder of the segmentation model.
            pretrained_weights (str): Dataset the encoder was trained on. Can be one of the following: "imagenet", "image-micronet", "micronet" or "None".
            normalization (dict): Parameters for the normalization.

    """

    def __init__(
        self,
        n_classes,
        learning_rate,
        architecture,
        encoder,
        pretrained_weights,
        normalization,
    ):
        super().__init__()
        model = pmm.segmentation_training.create_segmentation_model(
            architecture=architecture,
            encoder=encoder,
            encoder_weights=pretrained_weights,
            classes=n_classes,
        )
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = FocalTverskyLoss()
        self.f1_score = BinaryF1Score()
        self.normalization = normalization
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def log_tb_images(self, viz_batch) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "train_f1_score",
            f1_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "val_f1_score",
            f1_score,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        architecture,
        input_channels,
        n_classes,
        learning_rate,
        normalization,
        deep_supervision,
    ):
        """Pytorch Lightning Module for training a segmentation model.

        Parameters:
            architecture (str): The architecture of the segmentation model.
            input_channels (int): The number of input channels.
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            normalization (dict): Parameters for the normalization.
            deep_supervision (bool): Whether to use deep supervision or not.

        """

        super().__init__()
        if architecture == "Unet":
            model = Unet(num_classes=n_classes, input_channels=input_channels)
        elif architecture == "UnetPlusPlus":
            model = UnetPlusPlus(
                num_classes=n_classes,
                input_channels=input_channels,
                deep_supervision=deep_supervision,
            )
        else:
            raise ValueError(
                f"Architecture {architecture} not implemented. Implemented architectures are: Unet, UnetPlusPlus"
            )
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = FocalTverskyLoss()
        self.f1_score = BinaryF1Score()
        self.normalization = normalization
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def log_tb_images(self) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "train_f1_score",
            f1_score,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        f1_score = self.f1_score(y_hat, y)
        self.log(
            "val_f1_score",
            f1_score,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
