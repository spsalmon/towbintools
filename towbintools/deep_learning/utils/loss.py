from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-1,
        smooth=100,
        alpha=0.3,
        beta=0.7,
        gamma=4 / 3,
        activation=True,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.activation = activation

    def forward(self, inputs, targets):
        if self.activation:
            inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        smooth = self.smooth
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        # remove elements with ignore_index from the loss calculation
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky + 1e-10) ** gamma

        return FocalTversky


# addapted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class MultiClassFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # get probabilities for focal term calculation
        log_p = F.log_softmax(x, dim=-1)

        ce = self.ce_loss(log_p, y)

        y = y.long()
        # get true class column from each row
        log_pt = torch.gather(log_p, 1, y.unsqueeze(1)).squeeze(1)

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class BCELossWithIgnore(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction="none")
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


class PeakWeightedMSELoss(nn.Module):
    def __init__(self, ignore_index=-1, peak_weight=3.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.peak_weight = peak_weight

    def forward(self, input, target):
        weights = 1.0 + self.peak_weight * target  # Higher weight for higher values
        return (weights * (input - target) ** 2).mean()
