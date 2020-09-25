from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class RegHead(nn.Module):

    def __init__(self, feature_shape, in_channels, num_class):
        super().__init__()
        self.reg = nn.Linear(feature_shape[0] * feature_shape[1] * in_channels,
                             num_class)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        return self.softmax(self.reg(x.view(x.shape[0], -1)))
