import os
from math import ceil
from math import floor

import torch

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


# def change_first_conv_layer_input(model, new_in_channels):
#     for name, module in model.named_children():
#         if isinstance(module, Conv2dStaticSamePadding):
#             # Extract parameters from the original layer
#             out_channels = module.out_channels
#             kernel_size = module.kernel_size
#             stride = module.stride
#             dilation = module.dilation
#             groups = module.groups
#             bias = module.bias is not None

#             # Create a new Conv2d layer with the desired number of input channels
#             new_conv = Conv2dStaticSamePadding(
#                 new_in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,  # type: ignore
#                 stride=stride,
#                 dilation=dilation,  # type: ignore
#                 groups=groups,
#                 bias=bias,
#                 image_size=512,
#             )  # I think the image_size parameter can be anything but is required anyway

#             # Replace the original layer with the new one
#             setattr(model, name, new_conv)
#             break
#         if isinstance(module, nn.Conv2d):
#             # Extract parameters from the original layer
#             out_channels = module.out_channels
#             kernel_size = module.kernel_size
#             stride = module.stride
#             padding = module.padding
#             dilation = module.dilation
#             groups = module.groups
#             bias = module.bias is not None

#             # Create a new Conv2d layer with the desired number of input channels
#             new_conv = nn.Conv2d(
#                 new_in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,  # type: ignore
#                 stride=stride,
#                 padding=padding,
#                 dilation=dilation,  # type: ignore
#                 groups=groups,
#                 bias=bias,
#             )

#             # Replace the original layer with the new one
#             setattr(model, name, new_conv)
#             break

#         elif len(list(module.children())) > 0:
#             # Recursively call the function for nested modules (e.g., nn.Sequential)
#             change_first_conv_layer_input(module, new_in_channels)
#             break  # Break after modifying the first conv layer in any nested module


# def change_last_fc_layer_output(model, new_out_features):
#     # Reverse iterate through the model's children
#     for name, module in reversed(list(model.named_children())):
#         if isinstance(module, nn.Linear):
#             # Extract parameters from the original layer
#             in_features = module.in_features

#             # Create a new Linear layer with the desired output features
#             new_linear = nn.Linear(in_features, new_out_features)

#             # Replace the original layer with the new one
#             setattr(model, name, new_linear)
#             break
#         elif len(list(module.children())) > 0:
#             # Recursively call the function for nested modules (e.g., nn.Sequential)
#             change_last_fc_layer_output(module, new_out_features)
#             break  # Break after modifying the last linear layer in any nested module


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
    # Load the full checkpoint
    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")

    # Display original file size
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    print(f"Original file size: {original_size:.2f} MB")

    # Create lightweight checkpoint with only essential information
    lightweight_checkpoint = {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint.get("hyper_parameters", {}),
    }

    # Optionally keep other useful metadata
    optional_keys = ["epoch", "global_step", "pytorch-lightning_version"]
    for key in optional_keys:
        if key in checkpoint:
            lightweight_checkpoint[key] = checkpoint[key]

    # Save the lightweight version
    torch.save(lightweight_checkpoint, output_path)

    # Display new file size and savings
    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    savings = ((original_size - new_size) / original_size) * 100

    print(f"Lightweight file size: {new_size:.2f} MB")
    print(f"Space saved: {savings:.1f}%")

    # Show what was removed
    removed_keys = set(checkpoint.keys()) - set(lightweight_checkpoint.keys())
    if removed_keys:
        print(f"Removed keys: {removed_keys}")

    return lightweight_checkpoint
