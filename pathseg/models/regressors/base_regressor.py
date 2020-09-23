import torch.nn as nn

from ..builder import REGRESSORS, build_backbone, build_head


@REGRESSORS.register_module()
class BaseRegressor(nn.Module):

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)

    def forward(self, x):
        for stage in self.backbone.stages:
            x = stage(x)
        x = self.head(x)
        return x
