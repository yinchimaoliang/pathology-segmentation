import torch.nn as nn

from ..builder import ENCODERS, build_backbone


@ENCODERS.register_module()
class BaseEncoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = build_backbone(backbone)

    def forward(self, x):
        features = []
        for stage in self.backbone.stages:
            x = stage(x)
            features.insert(0, x)

        return features
