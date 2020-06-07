from pathseg.models import UnetDecoder, build_decoder


def test_unet_decoder():
    cfg = dict(
        type='UnetDecoder',
        encoder_channels=(512, 256, 128, 64, 64),
        decoder_channels=(512, 256, 128, 64, 64))

    unet_decoder = build_decoder(cfg)

    assert isinstance(unet_decoder, UnetDecoder)
