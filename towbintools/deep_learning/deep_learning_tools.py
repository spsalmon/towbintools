from .architectures.models import PretrainedSegmentationModel, SegmentationModel

def create_pretrained_segmentation_model(
    n_classes,
    architecture,
    encoder="efficientnet-b4",
    pretrained_weights="image-micronet",
    normalization={"type": "percentile", "lo": 1, "hi": 99},
    learning_rate=1e-5
):
    """Creates a segmentation model with a pretrained encoder.

    Parameters:
        n_classes (int): The number of classes in the segmentation task.
        learning_rate (float): The learning rate for the optimizer.
        architecture (str): The architecture of the segmentation model.
        encoder (str): The encoder of the segmentation model.
        pretrained_weights (str): Dataset the encoder was trained on. Can be one of the following: "imagenet", "image-micronet", "micronet" or "None".
        normalization (dict): Parameters for the normalization.

    Returns:
        PretrainedSegmentationModel: The segmentation model with a pretrained encoder.

    """
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
    learning_rate = 1e-5,
    deep_supervision=False,
):
    """Creates a segmentation model.

    Parameters:
        architecture (str): The architecture of the segmentation model.
        input_channels (int): The number of input channels.
        n_classes (int): The number of classes in the segmentation task.
        learning_rate (float): The learning rate for the optimizer.
        normalization (dict): Parameters for the normalization.
        deep_supervision (bool): Whether to use deep supervision or not.

    Returns:
        SegmentationModel: The segmentation model.

    """
    model = SegmentationModel(
        architecture=architecture,
        input_channels=input_channels,
        n_classes=n_classes,
        learning_rate=learning_rate,
        normalization=normalization,
        deep_supervision=deep_supervision,
    )
    return model