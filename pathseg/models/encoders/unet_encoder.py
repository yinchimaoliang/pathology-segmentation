import torch.nn as nn

from ..builder import ENCODERS, build_backbone


@ENCODERS.register_module()
class UnetEncoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = build_backbone(backbone)

    def forward(self, x):
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)

        x1 = self.backbone.maxpool(x0)
        x1 = self.backbone.layer1(x1)

        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        return [x4, x3, x2, x1, x0]
