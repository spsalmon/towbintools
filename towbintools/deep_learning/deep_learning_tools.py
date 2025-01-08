from towbintools.deep_learning.architectures import (
    PretrainedSegmentationModel,
    SegmentationModel,
)

from towbintools.deep_learning.utils.util import (
    get_input_channels_from_checkpoint
)

def create_pretrained_segmentation_model(
    input_channels = 3,
    n_classes=1,
    architecture="UnetPlusPlus",
    encoder="efficientnet-b4",
    pretrained_weights="image-micronet",
    normalization={"type": "percentile", "lo": 1, "hi": 99},
    learning_rate=1e-5,
    checkpoint_path=None,
    criterion=None,
):
    """Creates a segmentation model with a pretrained encoder.

    Parameters:
        n_classes (int): The number of classes in the segmentation task.
        learning_rate (float): The learning rate for the optimizer.
        architecture (str): The architecture of the segmentation model.
        encoder (str): The encoder of the segmentation model.
        pretrained_weights (str): Dataset the encoder was trained on. Can be one of the following: "imagenet", "image-micronet", "micronet" or "None".
        normalization (dict): Parameters for the normalization.
        checkpoint_path (str): Path to a checkpoint file.
        criterion (torch.nn.Module): The loss function.

    Returns:
        PretrainedSegmentationModel: The segmentation model with a pretrained encoder.

    """
    if checkpoint_path is not None:
        model = PretrainedSegmentationModel.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
        # change the learning rate and normalization
        model.learning_rate = learning_rate
        model.normalization = normalization
        return model

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
    return model


def create_segmentation_model(
    architecture,
    input_channels,
    n_classes,
    normalization={"type": "percentile", "lo": 1, "hi": 99},
    learning_rate=1e-5,
    deep_supervision=False,
    checkpoint_path=None,
    criterion=None,
):
    """Creates a segmentation model.

    Parameters:
        architecture (str): The architecture of the segmentation model.
        input_channels (int): The number of input channels.
        n_classes (int): The number of classes in the segmentation task.
        learning_rate (float): The learning rate for the optimizer.
        normalization (dict): Parameters for the normalization.
        deep_supervision (bool): Whether to use deep supervision or not.
        checkpoint_path (str): Path to a checkpoint file.
        criterion (torch.nn.Module): The loss function.

    Returns:
        SegmentationModel: The segmentation model.

    """

    if checkpoint_path is not None:
        model = SegmentationModel.load_from_checkpoint(checkpoint_path)
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

def load_pretrained_model_from_checkpoint(checkpoint_path):
    """Loads a pretrained segmentation model from a checkpoint. If the model cannot be loaded
    tries to load it with the default number of input channels.

    Parameters:
        checkpoint_path (str): Path to a checkpoint file.

    Returns:
        PretrainedSegmentationModel: The pretrained segmentation model.

    Raises:
        ValueError: If the model cannot be loaded from the checkpoint.
    """
    try:
        return PretrainedSegmentationModel.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        try:
            return PretrainedSegmentationModel.load_from_checkpoint(checkpoint_path, input_channels=get_input_channels_from_checkpoint(checkpoint_path))
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )

def load_scratch_segmentation_model_from_checkpoint(checkpoint_path, default_deep_supervision=False):
    """Loads a segmentation model from a checkpoint. If the model cannot be loaded
    tries to load it with the default number of input channels and deep supervision turned off.

    This function first tries to load the model as a PretrainedSegmentationModel. If that fails,
    it tries to load it as a SegmentationModel. If both attempts fail, it raises an error.

    Parameters:
        checkpoint_path (str): Path to a checkpoint file.

    Returns:
        SegmentationModel or PretrainedSegmentationModel: The segmentation model.

    Raises:
        ValueError: If the model cannot be loaded from the checkpoint.
    """
    try:
        return SegmentationModel.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        try:
            return SegmentationModel.load_from_checkpoint(checkpoint_path, input_channels=get_input_channels_from_checkpoint(checkpoint_path), deep_supervision=default_deep_supervision)
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )

def load_segmentation_model_from_checkpoint(checkpoint_path):
    """Loads a segmentation model from a checkpoint.

    This function first tries to load the model as a PretrainedSegmentationModel. If that fails,
    it tries to load it as a SegmentationModel. If both attempts fail, it raises an error.

    Parameters:
        checkpoint_path (str): Path to a checkpoint file.

    Returns:
        SegmentationModel or PretrainedSegmentationModel: The segmentation model.

    Raises:
        ValueError: If the model cannot be loaded from the checkpoint.
    """

    try:
        return load_pretrained_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        try:
            return load_scratch_segmentation_model_from_checkpoint(checkpoint_path)
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )
