import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import MulticlassF1Score

from .archs import AttentionUnet1D
from .archs import Unet
from .archs import Unet1D
from .archs import UnetPlusPlus
from .archs import UnetPlusPlus1D
from towbintools.deep_learning.utils.loss import FocalTverskyLoss
from towbintools.deep_learning.utils.loss import PeakWeightedMSELoss


class PretrainedClassificationModel(pl.LightningModule):
    """Pytorch Lightning Module for training a classification model with a pretrained weights. The model and weights are loaded from the pretrained_microscopy_models package.
    Because of the way they were pretrained, input images are required to have 3 channels.

    Parameters:
            architecture (str): The architecture of the classification model.
            input_channels (int): The number of input channels.
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            normalization (dict): Parameters for the normalization.
    """

    def __init__(
        self,
        architecture,
        input_channels,
        n_classes,
        learning_rate,
        normalization,
    ):
        super().__init__()
        model = timm.create_model(
            architecture,
            pretrained=True,
            num_classes=n_classes,
            in_chans=input_channels,
        )

        self.model = model
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        if n_classes == 2:
            self.criterion = nn.BCELoss()
            self.f1_score = BinaryF1Score()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.f1_score = MulticlassF1Score(num_classes=n_classes)

        self.normalization = normalization
        self.save_hyperparameters()

    def forward(self, x):
        if self.n_classes == 2:
            return torch.sigmoid(self.model(x))
        else:
            return torch.softmax(self.model(x), dim=1)

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
        y_hat = self.forward(x).squeeze()
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        y = y.to(torch.float)
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
        y_hat = self.forward(x).squeeze()
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        y = y.to(torch.float)
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
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class PretrainedSegmentationModel(pl.LightningModule):
    """Pytorch Lightning Module for training a segmentation model with a pretrained encoder. The encoder is loaded from the pretrained_microscopy_models package.
    This model automatically contains an activation layer at the end addapted to the number of classes.

    Parameters:
            input_channels (int): The number of input channels.
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            architecture (str): The architecture of the segmentation model.
            encoder (str): The encoder of the segmentation model.
            pretrained_weights (str): Dataset the encoder was trained on.
            normalization (dict): Parameters for the normalization.
            criterion (torch.nn.Module): The loss function to use for training. (default: FocalTverskyLoss)
            ignore_index (int): Index to ignore in the loss calculation and F1Score.

    """

    def __init__(
        self,
        input_channels,
        n_classes,
        learning_rate,
        architecture,
        encoder,
        pretrained_weights,
        normalization,
        criterion=None,
        ignore_index=None,
    ):
        super().__init__()

        if n_classes == 1:
            activation = "sigmoid"
        else:
            activation = "softmax"

        model = smp.create_model(
            arch=architecture,
            encoder_name=encoder,
            encoder_weights=pretrained_weights,
            in_channels=input_channels,
            classes=n_classes,
            activation=activation,
        )

        self.model = model
        self.learning_rate = learning_rate
        self.ignore_index = ignore_index

        if criterion is None:
            if n_classes == 1:
                self.criterion = FocalTverskyLoss(ignore_index=self.ignore_index)
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            self.criterion = criterion

        try:
            if n_classes == 1:
                self.f1_score = BinaryF1Score(ignore_index=self.criterion.ignore_index)
            else:
                self.f1_score = MulticlassF1Score(
                    num_classes=n_classes,
                    ignore_index=self.criterion.ignore_index,
                )
        except Exception as e:
            print(f"Criterion does not support ignore_index: {e}")
            if n_classes == 1:
                self.f1_score = BinaryF1Score()
            else:
                self.f1_score = MulticlassF1Score(num_classes=n_classes)

        self.normalization = normalization
        self.save_hyperparameters(ignore=["criterion"])

    def forward(self, x):
        return self.model(x)

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

    def predict_step(self, batch):
        x = batch

        pred = self.model(x)  # prediction already has softmax / sigmoid applied

        # binarize predictions
        if self.n_classes == 1:
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        return pred

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
        deep_supervision=False,
        criterion=None,
        ignore_index=None,
    ):
        """Pytorch Lightning Module for training a segmentation model.

        Parameters:
            architecture (str): The architecture of the segmentation model.
            input_channels (int): The number of input channels.
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            normalization (dict): Parameters for the normalization.
            deep_supervision (bool): Whether to use deep supervision or not.
            criterion (torch.nn.Module): The loss function to use for training. Default is FocalTverskyLoss.
            ignore_index (int): Index to ignore in the loss calculation and F1Score.

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
        self.ignore_index = ignore_index

        if criterion is None:
            if n_classes == 1:
                self.criterion = FocalTverskyLoss(ignore_index=self.ignore_index)
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            self.criterion = criterion

        try:
            if n_classes == 1:
                self.f1_score = BinaryF1Score(ignore_index=self.criterion.ignore_index)
            else:
                self.f1_score = MulticlassF1Score(
                    num_classes=n_classes,
                    ignore_index=self.criterion.ignore_index,
                )
        except Exception as e:
            print(f"Criterion does not support ignore_index: {e}")
            if n_classes == 1:
                self.f1_score = BinaryF1Score()
            else:
                self.f1_score = MulticlassF1Score(num_classes=n_classes)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        self.normalization = normalization
        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return self.activation(y)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        y_hat = self.activation(y_hat)
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
        y_hat = self.activation(y_hat)
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

    def predict_step(self, batch):
        x = batch
        pred = self.model(x)
        pred = self.activation(pred)

        # binarize predictions
        if self.n_classes == 1:
            pred = pred > 0.5
        else:
            pred = torch.argmax(pred, dim=1)

        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class KeypointDetection1DModel(pl.LightningModule):
    def __init__(
        self,
        input_channels,
        n_classes,
        learning_rate,
        architecture="UnetPlusPlus",
        activation="sigmoid",
        criterion=None,
    ):
        """Pytorch Lightning Module for 1D Keypoint Detection using a U-Net architecture.

        Parameters:
            input_channels (int): The number of input channels.
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            architecture (str): The architecture of the segmentation model. (default: "Unet")
            criterion (torch.nn.Module): The loss function to use for training. (default: nn.MSELoss)
            activation (str): The activation function to use at the end of the model. (default: "sigmoid")
        """

        super().__init__()

        if architecture == "Unet":
            model = Unet1D(num_classes=n_classes, input_channels=input_channels)
        elif architecture == "AttentionUnet":
            model = AttentionUnet1D(
                num_classes=n_classes, input_channels=input_channels
            )
        elif architecture == "UnetPlusPlus":
            model = UnetPlusPlus1D(num_classes=n_classes, input_channels=input_channels)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.model = model
        self.learning_rate = learning_rate

        # Set a default ignore_index if not present
        if not hasattr(self, "ignore_index"):
            self.ignore_index = -100

        if criterion is None:
            # self.criterion = nn.MSELoss()
            # self.criterion = nn.L1Loss()
            self.criterion = PeakWeightedMSELoss()

        else:
            self.criterion = criterion

        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return self.activation(y)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        y_hat = self.activation(y_hat)
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
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        y_hat = self.activation(y_hat)
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

    def predict_step(self, batch):
        x = batch
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
