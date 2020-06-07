from pathseg.models import UnetEncoder, build_encoder


def test_unet_encoder():

    backbone = dict(type='ResNet', name='resnet18', weights='imagenet')

    cfg = dict(type='UnetEncoder', backbone=backbone)
    unet_encoder = build_encoder(cfg)

    assert isinstance(unet_encoder, UnetEncoder)
