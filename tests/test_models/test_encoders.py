import torch

from pathseg.models import DeeplabV3PlusEncoder, UnetEncoder, build_encoder


def test_unet_encoder():

    backbone = dict(type='ResNet', name='resnet18', weights='imagenet')

    cfg = dict(type='UnetEncoder', backbone=backbone)
    unet_encoder = build_encoder(cfg)

    assert isinstance(unet_encoder, UnetEncoder)
    input_feature = torch.rand((1, 3, 512, 512))
    out_features = unet_encoder(input_feature)
    assert out_features[0].shape == torch.Size([1, 512, 16, 16])
    assert out_features[1].shape == torch.Size([1, 256, 32, 32])
    assert out_features[2].shape == torch.Size([1, 128, 64, 64])
    assert out_features[3].shape == torch.Size([1, 64, 128, 128])
    assert out_features[4].shape == torch.Size([1, 64, 256, 256])


def test_deeplabv3plus_encoder():
    backbone = dict(type='ResNet', name='resnet18', weights='imagenet')
    cfg = dict(
        type='DeeplabV3PlusEncoder',
        backbone=backbone,
        encoder_output_stride=16)
    deeplabv3plus_encoder = build_encoder(cfg)
    assert isinstance(deeplabv3plus_encoder, DeeplabV3PlusEncoder)
    assert deeplabv3plus_encoder.backbone.layer4[0].conv1.dilation == (2, 2)
    assert deeplabv3plus_encoder.backbone.layer4[0].conv2.dilation == (2, 2)

    input_feature = torch.rand((2, 3, 512, 512))
    out_features = deeplabv3plus_encoder(input_feature)
    assert out_features[0].shape == torch.Size([2, 256, 32, 32])
    assert out_features[1].shape == torch.Size([2, 64, 128, 128])
