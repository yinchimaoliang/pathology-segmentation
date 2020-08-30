import torch

from pathseg.models import build_segmenter
from pathseg.models.decoders import DeeplabV3PlusDecoder, UnetDecoder
from pathseg.models.encoders import DeeplabV3PlusEncoder, UnetEncoder


def test_unet():
    encoder = dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'))

    decoder = dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
        final_channels=2)

    cfg = dict(
        type='UNet', encoder=encoder, decoder=decoder, activation='softmax')

    unet = build_segmenter(cfg)

    assert isinstance(unet.encoder, UnetEncoder)
    assert isinstance(unet.decoder, UnetDecoder)

    input_features = torch.rand(2, 3, 256, 256)
    out_features = unet(input_features)
    assert out_features[0].shape == (2, 2, 256, 256)


def test_deeplabv3plus():
    deeplabv3plus_cfg = dict(
        type='DeeplabV3Plus',
        encoder=dict(
            type='DeeplabV3PlusEncoder',
            backbone=dict(type='ResNet', name='resnet18', weights='imagenet'),
            encoder_output_stride=16),
        decoder=dict(
            type='DeeplabV3PlusDecoder',
            encoder_channels=(256, 64),
            decoder_channels=[256, 48],
            output_stride=16,
            final_channels=2))

    deeplabv3plus = build_segmenter(deeplabv3plus_cfg)
    assert isinstance(deeplabv3plus.encoder, DeeplabV3PlusEncoder)
    assert isinstance(deeplabv3plus.decoder, DeeplabV3PlusDecoder)

    input_features = torch.rand(2, 3, 256, 256)
    out_features = deeplabv3plus(input_features)
    assert out_features.shape == (2, 2, 256, 256)


def test_nnunet():
    encoder = dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'))

    decoder = dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
        final_channels=2)

    cfg = dict(
        type='NNUNet', encoder=encoder, decoder=decoder, activation='softmax')

    nnunet = build_segmenter(cfg)
    assert isinstance(nnunet.encoder, UnetEncoder)
    assert isinstance(nnunet.decoder, UnetDecoder)
