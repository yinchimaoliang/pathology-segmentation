class_names = ['inflammation', 'low', 'high', 'carcinoma']
model = dict(
    type='BaseRegressor',
    backbone=dict(type='ResNet', name='resnet18', weights='imagenet'),
    head=dict(
        type='RegHead',
        feature_shape=(16, 16),
        in_channels=512,
        num_class=len(class_names)))

data = dict(
    class_names=class_names,
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='BaseDataset',
        data_root='./tests/data',
        classes=class_names,
        pipeline=[
            dict(type='Loading', shape=(512, 512), num_class=len(class_names)),
            # dict(
            #     type='Flip',
            #     prob=.5,
            #     flip_ratio_horizontal=.5,
            #     flip_ratio_vertical=.5),
            # dict(type='ShiftScaleRotate', prob=.5),
            # dict(type='RandomRotate90', prob=.5),
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=len(class_names) + 1)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512,
        use_path=True),
    valid=dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=len(class_names) + 1)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512),
    test=dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=len(class_names) + 1)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512,
    ))

train = dict(
    loss=dict(type='SmoothL1Loss'),
    optimizer=dict(type='Adam', lr=0.002, weight_decay=0.0001),
    scheduler=dict(step_size=10, gamma=0.1))

valid = dict(evals=['Dsc', 'Iou'])

test = dict(
    colors=[[0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0]],
    weight=0.2,
    evals=['Dsc', 'Iou'])

log_level = 'INFO'

dist_params = dict(backend='nccl')
