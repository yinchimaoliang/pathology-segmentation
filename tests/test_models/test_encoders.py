from pathseg.models import DeeplabV3PlusEncoder, UnetEncoder, build_encoder


def test_unet_encoder():

    backbone = dict(type='ResNet', name='resnet18', weights='imagenet')

    cfg = dict(type='UnetEncoder', backbone=backbone)
    unet_encoder = build_encoder(cfg)

    assert isinstance(unet_encoder, UnetEncoder)


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
