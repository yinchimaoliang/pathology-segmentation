import torch

from pathseg.models import build_head


def test_unet_head():
    conv_cfg = dict(type='Conv2d')
    norm_cfg = dict(type='BN', requires_grad=True)
    act_cfg = dict(type='ReLU')
    cfg = dict(
        type='UNetHead',
        in_channels=[64, 64, 128, 256, 512],
        channels=(256, 256, 128, 64, 64),
        dropout_ratio=0.1,
        num_classes=19,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        loss_decode=dict(type='BCEDiceLoss'))

    decode_head = build_head(cfg)
    input0 = torch.zeros([1, 64, 64, 64])
    input1 = torch.zeros([1, 64, 32, 32])
    input2 = torch.zeros([1, 128, 16, 16])
    input3 = torch.zeros([1, 256, 8, 8])
    input4 = torch.zeros([1, 512, 4, 4])
    inputs = [input0, input1, input2, input3, input4]
    outputs = decode_head(inputs)
    assert outputs.shape == torch.Size([1, 64, 128, 128])
