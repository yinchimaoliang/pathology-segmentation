import torch

from pathseg.models import DeepLabV3PlusDecoder, UnetDecoder, build_decoder


def test_unet_decoder():
    cfg = dict(
        type='UnetDecoder',
        encoder_channels=(512, 256, 128, 64, 64),
        decoder_channels=(512, 256, 128, 64, 64))

    unet_decoder = build_decoder(cfg)

    assert isinstance(unet_decoder, UnetDecoder)


def test_decoder():
    cfg = dict(
        type='DeepLabV3PlusDecoder',
        encoder_channels=(256, 64),
        output_stride=16,
        final_channels=2)

    deeplabv3plus_decoder = build_decoder(cfg)
    assert isinstance(deeplabv3plus_decoder, DeepLabV3PlusDecoder)
    assert deeplabv3plus_decoder.final_conv.in_channels == 256
    assert deeplabv3plus_decoder.final_conv.out_channels == 2

    features_0 = torch.rand(2, 256, 16, 16)
    features_1 = torch.rand(2, 64, 64, 64)

    final_features = deeplabv3plus_decoder([features_0, features_1])
    assert final_features.shape == (2, 2, 256, 256)
