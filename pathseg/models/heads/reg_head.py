import torch
from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class RegHead(nn.Module):

    def __init__(self, feature_shape, in_channels, num_class):
        super().__init__()
        self.reg = nn.Linear(feature_shape[0] * feature_shape[1] * in_channels,
                             num_class)

    def forward(self, x):
        return self.reg(torch.flatten(x))