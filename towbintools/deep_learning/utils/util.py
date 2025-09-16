import os
from math import ceil
from math import floor

import torch
from torch.serialization import safe_globals

# import torch.nn as nn
# from efficientnet_pytorch.utils import Conv2dStaticSamePadding


def divide_batch(batch, n):
    for i in range(0, batch.shape[0], n):
        yield batch[i : i + n, ::]  # noqa: E203


def get_closest_upper_multiple(dim, multiple):
    return int(multiple * ceil(dim / multiple))


def get_closest_lower_multiple(dim, multiple):
    return int(multiple * floor(dim / multiple))


def adjust_tensor_dimensions(source_tensor, target_tensor_shape):
    # Squeeze out unnecessary dimensions from source tensor
    adjusted_tensor = source_tensor.squeeze()
    # Add necessary dimensions to match the target tensor shape
    for dim, size in enumerate(target_tensor_shape):
        if dim >= len(adjusted_tensor.shape):
            adjusted_tensor = adjusted_tensor.unsqueeze(dim)
    return adjusted_tensor


def rename_keys_and_adjust_dimensions(model, pretrained_model):
    """Util function to easily load a pretrained model's weights into a model with the same architecture but different module names, etc."""
    assert len(model.state_dict().keys()) == len(
        pretrained_model.keys()
    ), f"The number of keys in the model ({len(model.state_dict().keys())}) and the pretrained model ({len(pretrained_model.keys())}) do not match."
    new_state_dict = {}
    for key, old_key in zip(model.state_dict().keys(), pretrained_model.keys()):
        target_tensor = model.state_dict()[key]
        source_tensor = pretrained_model[old_key]
        # Adjust the dimensions of the source tensor to match the target tensor
        if source_tensor.shape != target_tensor.shape:
            source_tensor = adjust_tensor_dimensions(source_tensor, target_tensor.shape)
        new_state_dict[key] = source_tensor
    return new_state_dict


def get_input_channels_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    input_channels = 0
    for key in state_dict.keys():
        if "conv" in key and "weight" in key:
            input_channels = state_dict[key].shape[1]
            break
    return input_channels


def create_lightweight_checkpoint(input_path, output_path):
    """
    Load existing PyTorch Lightning checkpoint and save a lightweight version
    """
    print(f"Loading checkpoint from: {input_path}")

    try:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Standard loading failed, trying with safe_globals: {e}")
        with safe_globals([getattr]):
            checkpoint = torch.load(input_path, map_location="cpu")

    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"Original file size: {original_size:.2f} MB")

    # Create lightweight checkpoint with only essential information
    lightweight_checkpoint = {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint.get("hyper_parameters", {}),
    }

    optional_keys = ["epoch", "global_step", "pytorch-lightning_version"]
    for key in optional_keys:
        if key in checkpoint:
            lightweight_checkpoint[key] = checkpoint[key]

    torch.save(lightweight_checkpoint, output_path)

    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    savings = ((original_size - new_size) / original_size) * 100

    print(f"Lightweight file size: {new_size:.2f} MB")
    print(f"Space saved: {savings:.1f}%")

    removed_keys = set(checkpoint.keys()) - set(lightweight_checkpoint.keys())
    if removed_keys:
        print(f"Removed keys: {removed_keys}")

    return lightweight_checkpoint
