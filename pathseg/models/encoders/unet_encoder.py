import torch.nn as nn

from ..builder import ENCODERS, build_backbone


@ENCODERS.register_module()
class UnetEncoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = build_backbone(backbone)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]
