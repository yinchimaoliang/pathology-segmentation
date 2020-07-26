import torch

from pathseg.models import DeeplabV3PlusDecoder, UnetDecoder, build_decoder


def test_unet_decoder():
    cfg = dict(
        type='UnetDecoder',
        encoder_channels=(512, 256, 128, 64, 64),
        decoder_channels=(512, 256, 128, 64, 64),
        final_channels=2)

    unet_decoder = build_decoder(cfg)

    assert isinstance(unet_decoder, UnetDecoder)

    feature_0 = torch.rand([2, 512, 8, 8])
    feature_1 = torch.rand([2, 256, 16, 16])
    feature_2 = torch.rand([2, 128, 32, 32])
    feature_3 = torch.rand([2, 64, 64, 64])
    feature_4 = torch.rand([2, 64, 128, 128])

    final_features = unet_decoder(
        [feature_0, feature_1, feature_2, feature_3, feature_4])
    assert final_features[0].shape == (2, 2, 256, 256)


def test_decoder():
    cfg = dict(
        type='DeeplabV3PlusDecoder',
        encoder_channels=(256, 64),
        output_stride=16,
        final_channels=2)

    deeplabv3plus_decoder = build_decoder(cfg)
    assert isinstance(deeplabv3plus_decoder, DeeplabV3PlusDecoder)
    assert deeplabv3plus_decoder.final_conv.in_channels == 256
    assert deeplabv3plus_decoder.final_conv.out_channels == 2

    features_0 = torch.rand(2, 256, 16, 16)
    features_1 = torch.rand(2, 64, 64, 64)

    final_features = deeplabv3plus_decoder([features_0, features_1])
    assert final_features[0].shape == (2, 2, 256, 256)
