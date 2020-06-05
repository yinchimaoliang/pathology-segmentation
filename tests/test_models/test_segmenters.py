from pathseg.models.decoders import UnetDecoder
from pathseg.models.encoders import UnetEncoder
from pathseg.models.segmenters import UNet


def test_unet():
    encoder = dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'))

    decoder = dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
    )

    unet = UNet(encoder, decoder, 'softmax')

    assert isinstance(unet.encoder, UnetEncoder)
    assert isinstance(unet.decoder, UnetDecoder)
