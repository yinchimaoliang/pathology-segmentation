import mmcv
import torch
import torchvision.transforms as tfs

from pathseg.models import ResNet, build_backbone


def test_resnet():
    norm_cfg = dict(type='BN', requires_grad=True)
    cfg = dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True)
    resnet = build_backbone(cfg)
    img = mmcv.imread('./tests/data/images/test.png')
    img = mmcv.imresize(img, (128, 128))
    im_tfs = tfs.Compose([
        tfs.ToTensor(),  # [0-255]--->[0-1]
        tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    ])
    img = im_tfs(img)
    img = torch.unsqueeze(img, 0)
    outputs = resnet(img)
    assert outputs[0].shape == torch.Size([1, 256, 32, 32])
    assert outputs[1].shape == torch.Size([1, 512, 16, 16])
    assert outputs[2].shape == torch.Size([1, 1024, 8, 8])
    assert outputs[3].shape == torch.Size([1, 2048, 4, 4])
    assert isinstance(resnet, ResNet)
