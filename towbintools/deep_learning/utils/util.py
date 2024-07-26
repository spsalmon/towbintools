import torch.nn as nn

def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i : i + n, ::]

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
    assert len(model.state_dict().keys()) == len(pretrained_model.keys()), f"The number of keys in the model ({len(model.state_dict().keys())}) and the pretrained model ({len(pretrained_model.keys())}) do not match."
    new_state_dict = {}
    for key, old_key in zip(model.state_dict().keys(), pretrained_model.keys()):
        target_tensor = model.state_dict()[key]
        source_tensor = pretrained_model[old_key]
        # Adjust the dimensions of the source tensor to match the target tensor
        if source_tensor.shape != target_tensor.shape:
            source_tensor = adjust_tensor_dimensions(source_tensor, target_tensor.shape)
        new_state_dict[key] = source_tensor
    return new_state_dict

def change_first_conv_layer_input(model, new_in_channels):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # Extract parameters from the original layer
            out_channels = module.out_channels
            kernel_size = module.kernel_size
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups
            bias = module.bias is not None
            
            # Create a new Conv2d layer with the desired number of input channels
            new_conv = nn.Conv2d(new_in_channels, out_channels, kernel_size=kernel_size, # type: ignore
                                 stride=stride, padding=padding, dilation=dilation,  # type: ignore
                                 groups=groups, bias=bias)
            
            # Replace the original layer with the new one
            setattr(model, name, new_conv)
            break
        elif len(list(module.children())) > 0:
            # Recursively call the function for nested modules (e.g., nn.Sequential)
            change_first_conv_layer_input(module, new_in_channels)
            break  # Break after modifying the first conv layer in any nested module

def change_last_fc_layer_output(model, new_out_features):
    # Reverse iterate through the model's children
    for name, module in reversed(list(model.named_children())):
        if isinstance(module, nn.Linear):
            # Extract parameters from the original layer
            in_features = module.in_features
            
            # Create a new Linear layer with the desired output features
            new_linear = nn.Linear(in_features, new_out_features)
            
            # Replace the original layer with the new one
            setattr(model, name, new_linear)
            break
        elif len(list(module.children())) > 0:
            # Recursively call the function for nested modules (e.g., nn.Sequential)
            change_last_fc_layer_output(module, new_out_features)
            break  # Break after modifying the last linear layer in any nested module