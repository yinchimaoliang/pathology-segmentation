conv_cfg = dict(type='Conv2d')
norm_cfg = dict(type='BN', requires_grad=True)
act_cfg = dict(type='ReLU')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_cfg = dict(mode='slide', stride=[512, 512], crop_size=[512, 512])
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3, 4),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='UNetHead',
        in_channels=[64, 64, 128, 256, 512],
        channels=(256, 256, 128, 64, 64),
        dropout_ratio=0.1,
        num_classes=5,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=test_cfg)

data = dict(
    class_names=['background', 'Inflammation', 'Low', 'High', 'Carcinoma'],
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='BaseDataset',
        data_root='./data/train',
        pipeline=[
            dict(type='RandomCrop', crop_size=[512, 512], cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ],
        use_patch=True,
        random_sampling=True,
        repeat=100,
        classes=['background', 'Inflammation', 'Low', 'High', 'Carcinoma']),
    val=dict(
        type='BaseDataset',
        data_root='./data/valid',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ]),
        ],
        use_patch=False,
        random_sampling=False,
        classes=['background', 'Inflammation', 'Low', 'High', 'Carcinoma']),
    test=dict(
        type='BaseDataset',
        data_root='./data/valid',
        pipeline=[
            dict(type='LoadPatch'),
            dict(type='RandomCrop', crop_size=[512, 512], cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ],
        use_patch=True,
        random_sampling=True,
        repeat=100,
        classes=['background', 'Inflammation', 'Low', 'High', 'Carcinoma']))

train = dict(
    loss=dict(
        type='BCEDiceLoss', reduction='none', pos_weight=[0.1, 1, 10, 10,
                                                          100]),
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001),
    scheduler=dict(step_size=10, gamma=0.1))

valid = dict(evals=['Dsc', 'Iou'])

test = dict(
    colors=[[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]],
    weight=0.2,
    evals=['Dsc', 'Iou'])

log_level = 'INFO'

dist_params = dict(backend='nccl')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=200, metric='mIoU')
train_cfg = dict()

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
