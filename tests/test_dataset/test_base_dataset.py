import numpy as np
import torch

from pathseg.datasets import BaseDataset, build_dataset


def test_base_dataset():
    np.random.seed(0)
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    crop_size = (769, 769)
    pipelines = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg = dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=pipelines,
        use_patch=False)

    base_dataset = build_dataset(cfg)
    assert isinstance(base_dataset, BaseDataset)

    sample = base_dataset[0]

    img = sample['img']
    gt_semantic_seg = sample['gt_semantic_seg']

    assert img.data.shape == torch.Size([3, 769, 769])
    assert gt_semantic_seg.data.shape == torch.Size([1, 769, 769])
