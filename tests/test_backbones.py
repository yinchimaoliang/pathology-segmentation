import torch.nn as nn

from pathseg.models.backbones import ResNet


def test_resnet():

    resnet = ResNet('resnet18', 'imagenet')
    assert isinstance(resnet.layer1[0].conv1, nn.modules.conv.Conv2d)
