from pathseg.core.common.blocks import Conv2dReLU
from pathseg.models.decoders import UnetDecoder


def test_unet_decoder():
    unet_decoder = UnetDecoder((512, 256, 128, 64, 64),
                               (512, 256, 128, 64, 64))

    assert isinstance(unet_decoder.layer1.block[0], Conv2dReLU)
