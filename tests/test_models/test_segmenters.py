from pathseg.models import build_segmenter
from pathseg.models.decoders import UnetDecoder
from pathseg.models.encoders import UnetEncoder


def test_unet():
    encoder = dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'))

    decoder = dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
    )

    cfg = dict(
        type='UNet', encoder=encoder, decoder=decoder, activation='softmax')

    unet = build_segmenter(cfg)

    assert isinstance(unet.encoder, UnetEncoder)
    assert isinstance(unet.decoder, UnetDecoder)
