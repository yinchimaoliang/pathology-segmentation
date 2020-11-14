import mmcv
import torch
import torchvision.transforms as tfs

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


def test_encoder_decoder():
    conv_cfg = dict(type='Conv2d')
    norm_cfg = dict(type='BN', requires_grad=True)
    act_cfg = dict(type='ReLU')
    backbone = dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True)

    unet_head = dict(
        type='UNetHead',
        in_channels=[64, 64, 128, 256, 512],
        channels=(256, 256, 128, 64, 64),
        dropout_ratio=0.1,
        num_classes=19,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        loss_decode=dict(type='BCEDiceLoss'))

    test_cfg = mmcv.Config(dict(mode='whole'))
    cfg = dict(
        type='EncoderDecoder',
        backbone=backbone,
        decode_head=unet_head,
        test_cfg=test_cfg)

    encoder_decoder = build_segmenter(cfg)

    img = mmcv.imread('./tests/data/images/test.png')
    img = mmcv.imresize(img, (128, 128))
    im_tfs = tfs.Compose([
        tfs.ToTensor(),  # [0-255]--->[0-1]
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    ])
    img = im_tfs(img)
    img = torch.unsqueeze(img, 0)

    outputs = encoder_decoder([img], [[
        dict(
            ori_shape=[128, 128],
            img_shape=[128, 128],
            pad_shape=[0, 0],
            flip=False)
    ]],
                              return_loss=False)

    assert outputs[0].shape == torch.Size([128, 128])
