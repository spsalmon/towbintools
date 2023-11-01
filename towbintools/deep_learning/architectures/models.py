import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score
from towbintools.deep_learning.utils.loss import FocalTverskyLoss
import pretrained_microscopy_models as pmm

class LightningPretrained(pl.LightningModule):
	def __init__(self, n_classes, learning_rate, architecture, encoder, pretrained_weights, normalization):
		super().__init__()
		model = pmm.segmentation_training.create_segmentation_model(
			architecture=architecture,
			encoder = encoder,
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
				raise ValueError('TensorBoard Logger not found')

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		f1_score = self.f1_score(y_hat, y)
		self.log("train_f1_score", f1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		f1_score = self.f1_score(y_hat, y)
		self.log("val_f1_score", f1_score, on_step=True, on_epoch=True, logger=True, sync_dist=True)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer