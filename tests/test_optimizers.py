from torch.optim import SGD

from pathseg.core.optimizers.builder import build_optimizer
from pathseg.models.segmenters import UNet


def test_sgd():
    encoder = dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'))

    decoder = dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
    )

    unet = UNet(encoder, decoder, 'softmax')
    optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    # optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    sgd = build_optimizer(unet, optimizer)
    assert isinstance(sgd, SGD)
