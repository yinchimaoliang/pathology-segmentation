import torch
from torch import nn

from ..builder import LOSSES


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, weight=None):
        super().__init__()
        assert beta > 0
        self.beta = beta
        self.weight = weight

    def forward(self, pred, target):

        assert pred.size() == target.size() and target.numel() > 0
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                           diff - 0.5 * self.beta)
        if self.weight:
            return torch.sum(loss * self.weight)
        else:
            return torch.sum(loss)
