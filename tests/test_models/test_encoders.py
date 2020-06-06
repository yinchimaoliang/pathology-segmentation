from pathseg.models import UnetEncoder


def test_unet_encoder():

    backbone = dict(type='ResNet', name='resnet18', weights='imagenet')

    unet_encoder = UnetEncoder(backbone)

    print(unet_encoder)
