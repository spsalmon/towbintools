from typing import Any

from towbintools.deep_learning.architectures import KeypointDetection1DModel
from towbintools.deep_learning.architectures import PretrainedClassificationModel
from towbintools.deep_learning.architectures import PretrainedSegmentationModel
from towbintools.deep_learning.architectures import SegmentationModel
from towbintools.deep_learning.utils.util import (
    get_input_channels_from_checkpoint,
)


def create_classification_model(
    architecture: str,
    input_channels: int,
    classes: list[str],
    learning_rate: float = 1e-4,
    checkpoint_path: str | None = None,
    normalization: dict = {"type": "percentile", "lo": 1, "hi": 99},
) -> PretrainedClassificationModel:
    """
    Create a classification model.

    Parameters:
        architecture (str): ``timm`` model name (e.g. ``"efficientnet_b0"``).
        input_channels (int): Number of input image channels.
        classes (list[str]): List of class label strings.
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-4)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint; if
            provided, the model is loaded from the checkpoint and all other
            arguments are ignored. (default: None)
        normalization (dict, optional): Normalization config stored as a
            hyperparameter for inference. (default: percentile 1–99)

    Returns:
        PretrainedClassificationModel: Constructed or loaded classification model.
    """

    if checkpoint_path is not None:
        model = PretrainedClassificationModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
        return model

    model = PretrainedClassificationModel(
        architecture,
        input_channels,
        classes,
        learning_rate,
        normalization,
    )
    return model


def create_pretrained_segmentation_model(
    input_channels: int = 1,
    n_classes: int = 1,
    architecture: str = "UnetPlusPlus",
    encoder: str = "efficientnet-b4",
    pretrained_weights: str = "image-micronet",
    normalization: dict = {"type": "percentile", "lo": 1, "hi": 99},
    learning_rate: float = 1e-5,
    checkpoint_path: str | None = None,
    reset_optimizer: bool = True,
    criterion: Any | None = None,
) -> PretrainedSegmentationModel:
    """
    Create a segmentation model with a pretrained encoder.

    Parameters:
        input_channels (int, optional): Number of input image channels. (default: 1)
        n_classes (int, optional): Number of foreground segmentation classes.
            (default: 1)
        architecture (str, optional): ``smp`` architecture name. (default: ``"UnetPlusPlus"``)
        encoder (str, optional): Encoder backbone name. (default: ``"efficientnet-b4"``)
        pretrained_weights (str, optional): Dataset the encoder was pretrained on.
            (default: ``"image-micronet"``)
        normalization (dict, optional): Normalization config. (default: percentile 1–99)
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-5)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint. If
            provided, the model is loaded from the checkpoint, then
            ``learning_rate`` and ``normalization`` are updated. (default: None)
        reset_optimizer (bool, optional): If ``True``, discard the optimizer
            state from the checkpoint. (default: True)
        criterion (nn.Module, optional): Loss function. If ``None``, a default
            is chosen based on ``n_classes``. (default: None)

    Returns:
        PretrainedSegmentationModel: Constructed or loaded segmentation model.

    Raises:
        ValueError: If the checkpoint architecture or encoder does not match
            the requested one.
    """
    model = PretrainedSegmentationModel(
        input_channels=input_channels,
        n_classes=n_classes,
        learning_rate=learning_rate,
        architecture=architecture,
        encoder=encoder,
        pretrained_weights=pretrained_weights,
        normalization=normalization,
        criterion=criterion,
    )

    if checkpoint_path is not None:
        loaded_model = PretrainedSegmentationModel.load_from_checkpoint(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        if reset_optimizer:
            loaded_model.optimizer = None
            loaded_model.lr_scheduler = None

            loaded_model._optimizer = None
            loaded_model._lr_scheduler = None

        # change the learning rate and normalization
        loaded_model.learning_rate = learning_rate
        loaded_model.normalization = normalization

        # check if the architecture matches
        if not isinstance(loaded_model.model, model.model.__class__):
            raise ValueError(
                f"Checkpoint architecture {loaded_model.model.__class__} does not match the requested architecture {architecture}"
            )
        # check if the encoder matches
        if loaded_model.model.encoder.__class__ != model.model.encoder.__class__:
            raise ValueError(
                f"Checkpoint encoder architecture {type(loaded_model.model.encoder).__name__} does not match the requested encoder architecture {type(model.model.encoder).__name__}"
            )

        loaded_model.configure_optimizers()

        return loaded_model

    return model


def create_segmentation_model(
    architecture: str,
    input_channels: int,
    n_classes: int,
    normalization: dict = {"type": "percentile", "lo": 1, "hi": 99},
    learning_rate: float = 1e-5,
    deep_supervision: bool = False,
    checkpoint_path: str | None = None,
    reset_optimizer: bool = True,
    criterion: Any | None = None,
) -> SegmentationModel:
    """
    Create a segmentation model using a custom (non-pretrained) architecture.

    Parameters:
        architecture (str): Architecture name; one of ``"Unet"`` or
            ``"UnetPlusPlus"``.
        input_channels (int): Number of input image channels.
        n_classes (int): Number of foreground segmentation classes.
        normalization (dict, optional): Normalization config. (default: percentile 1–99)
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-5)
        deep_supervision (bool, optional): Enable deep supervision (only relevant
            for ``"UnetPlusPlus"``). (default: False)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint. If
            provided, the model is loaded from the checkpoint, then
            ``learning_rate``, ``normalization``, and ``deep_supervision`` are
            updated. (default: None)
        reset_optimizer (bool, optional): If ``True``, discard the optimizer
            state from the checkpoint. (default: True)
        criterion (nn.Module, optional): Loss function. If ``None``, a default
            is chosen based on ``n_classes``. (default: None)

    Returns:
        SegmentationModel: Constructed or loaded segmentation model.
    """

    if checkpoint_path is not None:
        model = SegmentationModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )

        if reset_optimizer:
            model.optimizer = None
            model.lr_scheduler = None
        model.learning_rate = learning_rate
        model.normalization = normalization
        model.deep_supervision = deep_supervision
        return model

    model = SegmentationModel(
        architecture=architecture,
        input_channels=input_channels,
        n_classes=n_classes,
        learning_rate=learning_rate,
        normalization=normalization,
        deep_supervision=deep_supervision,
        criterion=criterion,
    )

    return model


def create_keypoint_detection_model(
    architecture: str,
    input_channels: int,
    n_classes: int,
    learning_rate: float = 1e-4,
    checkpoint_path: str | None = None,
    criterion: Any | None = None,
    activation: str = "relu",
) -> KeypointDetection1DModel:
    """
    Create a 1D keypoint detection model.

    Parameters:
        architecture (str): Architecture name; one of ``"Unet"``,
            ``"AttentionUnet"``, or ``"UnetPlusPlus"``.
        input_channels (int): Number of input sequence channels.
        n_classes (int): Number of keypoint classes (output channels).
        learning_rate (float, optional): Learning rate for the Adam optimizer.
            (default: 1e-4)
        checkpoint_path (str, optional): Path to a ``.ckpt`` checkpoint; if
            provided, the model is loaded from the checkpoint and all other
            arguments are ignored. (default: None)
        criterion (nn.Module, optional): Loss function. If ``None``,
            ``PeakWeightedMSELoss`` is used. (default: None)
        activation (str, optional): Output activation; one of ``"relu"``,
            ``"leaky_relu"``, ``"sigmoid"``, or ``"none"``. (default: ``"relu"``)

    Returns:
        KeypointDetection1DModel: Constructed or loaded keypoint detection model.
    """

    if checkpoint_path is not None:
        model = KeypointDetection1DModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
        return model

    model = KeypointDetection1DModel(
        input_channels,
        n_classes,
        learning_rate,
        architecture=architecture,
        activation=activation,
        criterion=criterion,
    )
    return model


def load_pretrained_segmentation_model_from_checkpoint(
    checkpoint_path: str,
) -> PretrainedSegmentationModel:
    """
    Load a pretrained segmentation model from a checkpoint.

    First tries a direct load; if that fails (e.g. mismatched ``input_channels``
    in the checkpoint metadata), infers the channel count from the checkpoint
    weights and retries with ``pretrained_weights=None``.

    Parameters:
        checkpoint_path (str): Path to a ``.ckpt`` checkpoint file.

    Returns:
        PretrainedSegmentationModel: Loaded segmentation model.

    Raises:
        ValueError: If both loading attempts fail.
    """
    try:
        return PretrainedSegmentationModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    except Exception as e:
        try:
            return PretrainedSegmentationModel.load_from_checkpoint(
                checkpoint_path,
                input_channels=get_input_channels_from_checkpoint(checkpoint_path),
                pretrained_weights=None,
                weights_only=False,
            )
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )


def load_scratch_segmentation_model_from_checkpoint(
    checkpoint_path: str,
    default_deep_supervision: bool = False,
) -> SegmentationModel:
    """
    Load a :class:`SegmentationModel` from a checkpoint.

    First tries a direct load; if that fails, infers the channel count from
    the checkpoint weights and retries with ``deep_supervision=default_deep_supervision``.

    Parameters:
        checkpoint_path (str): Path to a ``.ckpt`` checkpoint file.
        default_deep_supervision (bool, optional): Deep supervision value used
            on the fallback load attempt. (default: False)

    Returns:
        SegmentationModel: Loaded segmentation model.

    Raises:
        ValueError: If both loading attempts fail.
    """
    try:
        return SegmentationModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    except Exception as e:
        try:
            return SegmentationModel.load_from_checkpoint(
                checkpoint_path,
                input_channels=get_input_channels_from_checkpoint(checkpoint_path),
                deep_supervision=default_deep_supervision,
                weights_only=False,
            )
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )


def load_segmentation_model_from_checkpoint(
    checkpoint_path: str,
) -> PretrainedSegmentationModel | SegmentationModel:
    """
    Load a segmentation model from a checkpoint, trying both model types.

    First tries to load as :class:`PretrainedSegmentationModel`; if that fails,
    tries :class:`SegmentationModel`. Raises if both fail.

    Parameters:
        checkpoint_path (str): Path to a ``.ckpt`` checkpoint file.

    Returns:
        PretrainedSegmentationModel or SegmentationModel: Loaded segmentation model.

    Raises:
        ValueError: If both loading attempts fail.
    """

    try:
        return load_pretrained_segmentation_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        try:
            return load_scratch_segmentation_model_from_checkpoint(checkpoint_path)
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )


def load_keypoint_detection_model_from_checkpoint(
    checkpoint_path: str,
) -> KeypointDetection1DModel:
    """
    Load a 1D keypoint detection model from a checkpoint.

    Parameters:
        checkpoint_path (str): Path to a ``.ckpt`` checkpoint file.

    Returns:
        KeypointDetection1DModel: Loaded keypoint detection model.

    Raises:
        ValueError: If the model cannot be loaded from the checkpoint.
    """
    try:
        return KeypointDetection1DModel.load_from_checkpoint(
            checkpoint_path, weights_only=False
        )
    except Exception as e:
        raise ValueError(
            f"Could not load keypoint detection model from checkpoint {checkpoint_path}. Error: {e}"
        )
