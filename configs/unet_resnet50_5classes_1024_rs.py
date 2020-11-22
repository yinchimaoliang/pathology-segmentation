model = dict(
    type='UNet',
    encoder=dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'),
    ),
    decoder=dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
        final_channels=5))

data = dict(
    class_names=['Inflammation', 'Low', 'High', 'Carcinoma'],
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='BaseDataset',
        data_root='./data/train',
        pipeline=[
            dict(
                type='RandomSampling',
                prob_global=.1,
                target_shape=(1024, 1024),
                filter_classes=[0]),
            dict(
                type='Flip',
                prob=.5,
                flip_ratio_horizontal=.5,
                flip_ratio_vertical=.5),
            dict(type='ShiftScaleRotate', prob=.5),
            dict(type='RandomRotate90', prob=.5),
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=5)
        ],
        use_patch=True,
        random_sampling=True,
        repeat=100,
        scale=1 / 4),
    valid=dict(
        type='BaseDataset',
        data_root='./data/valid',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=5)
        ],
        use_patch=True,
        random_sampling=False,
        width=1024,
        height=1024,
        stride=1024,
        scale=1 / 4),
    test=dict(
        type='BaseDataset',
        data_root='./data/valid',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=5)
        ],
        random_sampling=False,
        use_patch=True,
        width=1024,
        height=1024,
        stride=1024,
    ))

train = dict(
    loss=dict(type='BCEDiceLoss', reduction='none'),
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0001),
    scheduler=dict(step_size=10, gamma=0.1))

valid = dict(evals=['Dsc', 'Iou'])

test = dict(
    colors=[[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]],
    weight=0.2,
    evals=['Dsc', 'Iou'])

log_level = 'INFO'

dist_params = dict(backend='nccl')
