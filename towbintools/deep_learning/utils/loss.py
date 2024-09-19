import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(FocalTverskyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, smooth=100, alpha=0.3, beta=0.7, gamma=4 / 3):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # remove elements with ignore_index from the loss calculation
        mask = (targets != self.ignore_index)
        inputs = inputs[mask]
        targets = targets[mask]

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky + 1e-10) ** gamma

        return FocalTversky

class BCELossWithIgnore(nn.Module):
    def __init__(self, ignore_index=-1):
        super(BCELossWithIgnore, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Calculate BCE loss

        # Create a mask for non-ignored values
        mask = (target != self.ignore_index).float()

        # Set the ignored values to 0
        target = target * mask

        # Calculate the loss
        loss = self.bce_loss(input, target)

        # Apply the mask to the loss
        masked_loss = loss * mask
        
        # Return the mean of the masked loss
        return masked_loss.sum() / mask.sum().clamp(min=1e-8)
