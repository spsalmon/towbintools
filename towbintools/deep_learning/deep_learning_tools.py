from towbintools.deep_learning.architectures import (
    PretrainedSegmentationModel,
    SegmentationModel,
)


def create_pretrained_segmentation_model(
    n_classes=1,
    architecture="UnetPlusPlus",
    encoder="efficientnet-b4",
    pretrained_weights="image-micronet",
    normalization={"type": "percentile", "lo": 1, "hi": 99},
    learning_rate=1e-5,
    checkpoint_path=None,
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
        n_classes=n_classes,
        learning_rate=learning_rate,
        architecture=architecture,
        encoder=encoder,
        pretrained_weights=pretrained_weights,
        normalization=normalization,
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
    )

    return model


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
        return PretrainedSegmentationModel.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        try:
            return SegmentationModel.load_from_checkpoint(checkpoint_path)
        except Exception as e2:
            raise ValueError(
                f"Could not load model from checkpoint {checkpoint_path}. Error: {e} and {e2}"
            )
