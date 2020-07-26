model = dict(
    type='UNet',
    encoder=dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'),
    ),
    decoder=dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
        final_channels=2))

data = dict(
    class_names=['Lesions'],
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type='BaseDataset',
        data_root='./data/train',
        pipeline=[
            dict(
                type='RandomSampling', prob_global=.5,
                target_shape=(512, 512)),
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
                num_classes=2)
        ],
        random_sampling=True,
        width=512,
        height=512,
        stride=512,
        repeat=400),
    valid=dict(
        type='BaseDataset',
        data_root='./data/valid',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=2)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512,
    ),
    test=dict(
        type='BaseDataset',
        data_root='./data/test',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=2)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512,
    ))

train = dict(
    loss=dict(type='BCEDiceLoss'),
    optimizer=dict(type='Adam', lr=0.01, weight_decay=0.0001),
    scheduler=dict(step_size=30, gamma=0.1))

valid = dict(evals=['Dsc', 'Iou'])

test = dict(colors=[[0, 255, 0]], weight=0.2, evals=['Dsc', 'Iou'])

log_level = 'INFO'

dist_params = dict(backend='nccl')
