import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score
from towbintools.deep_learning.utils.loss import FocalTverskyLoss
from towbintools.deep_learning.utils.util import (change_first_conv_layer_input, change_last_fc_layer_output, rename_keys_and_adjust_dimensions)
import pretrained_microscopy_models as pmm
from .archs import Unet, UnetPlusPlus
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score
import timm
import torch.utils.model_zoo as model_zoo

PMM_TO_TIMM = {
    'densenet121': 'densenet121',
    'densenet161': 'densenet161', 
    'densenet169': 'densenet169',
    'densenet201': 'densenet201',
    'dpn107': 'dpn107',
    'dpn131': 'dpn131',
    'dpn68': 'dpn68',
    'dpn68b': 'dpn68b', 
    'dpn92': 'dpn92',
    'dpn98': 'dpn98',
    'efficientnet-b0': 'efficientnet_b0',
    'efficientnet-b1': 'efficientnet_b1',
    'efficientnet-b2': 'efficientnet_b2',
    'efficientnet-b3': 'efficientnet_b3',
    'efficientnet-b4': 'efficientnet_b4',
    'efficientnet-b5': 'efficientnet_b5',
    'efficientnet-b6': 'efficientnet_b6',
    'efficientnet-b7': 'efficientnet_b7',
    'inceptionresnetv2': 'inception_resnet_v2',
    'inceptionv4': 'inception_v4',
    'mobilenet_v2': 'mobilenetv2_100', 
    'resnet101': 'resnet101',
    'resnet152': 'resnet152',
    'resnet18': 'resnet18',
    'resnet34': 'resnet34',
    'resnet50': 'resnet50', 
    'resnext101_32x8d': 'ig_resnext101_32x8d',
    'resnext50_32x4d': 'resnext50_32x4d',
    'se_resnet101': 'seresnet101',
    'se_resnet152': 'seresnet152',
    'se_resnet50': 'seresnet50',
    'se_resnext101_32x4d': 'seresnext101_32x4d',
    'se_resnext50_32x4d': 'seresnext50_32x4d',
    'senet154': 'senet154',
    'vgg11_bn': 'vgg11_bn', 
    'vgg11': 'vgg11',
    'vgg13_bn': 'vgg13_bn',
    'vgg13': 'vgg13',
    'vgg16_bn': 'vgg16_bn',
    'xception': 'xception'
}

class PretrainedClassificationModel(pl.LightningModule):
    """Pytorch Lightning Module for training a classification model with a pretrained weights. The model and weights are loaded from the pretrained_microscopy_models package.
    Because of the way they were pretrained, input images are required to have 3 channels.

    Parameters:
            n_classes (int): The number of classes in the segmentation task.
            learning_rate (float): The learning rate for the optimizer.
            architecture (str): The architecture of the classification model.
            pretrained_weights (str): Dataset the encoder was trained on. Can be one of the following: "imagenet", "image-micronet", "micronet" or "None".
            normalization (dict): Parameters for the normalization.
    """

    def __init__(
        self,
        input_channels,
        n_classes,
        learning_rate,
        architecture,
        pretrained_weights,
        normalization,
    ):
        super().__init__()
        architecture_timm = PMM_TO_TIMM[architecture]
        model = timm.create_model(architecture_timm, pretrained=False)
        url = pmm.util.get_pretrained_microscopynet_url(architecture, pretrained_weights)
        pretrained_model = model_zoo.load_url(url)
        pretrained_model = rename_keys_and_adjust_dimensions(model, pretrained_model)
        model.load_state_dict(pretrained_model)

        if input_channels != 3:
            change_first_conv_layer_input(model, input_channels)

        if n_classes == 2:
            change_last_fc_layer_output(model, 1)
        else:
            change_last_fc_layer_output(model, n_classes)

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
            pretrained_weights (str): Dataset the encoder was trained on. Can be one of the following: "imagenet", "image-micronet", "micronet" or "None".
            normalization (dict): Parameters for the normalization.
            criterion (torch.nn.Module): The loss function to use for training. Default is FocalTverskyLoss.
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
        criterion = None,
        ignore_index=None,
    ):
        super().__init__()
        model = pmm.segmentation_training.create_segmentation_model(
            architecture=architecture,
            encoder=encoder,
            encoder_weights=pretrained_weights,
            classes=n_classes,
        )

        if input_channels != 3:
            change_first_conv_layer_input(model, input_channels)

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
                self.f1_score = MulticlassF1Score(num_classes=n_classes, ignore_index=self.criterion.ignore_index)
        except Exception as e:
            print(f'Criterion does not support ignore_index: {e}')
            if n_classes == 1:
                self.f1_score = BinaryF1Score()
            else:
                self.f1_score = MulticlassF1Score(num_classes=n_classes)

        self.normalization = normalization
        self.save_hyperparameters(ignore=["criterion"])

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

    def predict_step(self, batch):
        x = batch
        
        pred = self.model(x) # prediction already has softmax / sigmoid applied

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
        deep_supervision,
        criterion = None,
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
                self.f1_score = MulticlassF1Score(num_classes=n_classes, ignore_index=self.criterion.ignore_index)
        except Exception as e:
            print(f'Criterion does not support ignore_index: {e}')
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
