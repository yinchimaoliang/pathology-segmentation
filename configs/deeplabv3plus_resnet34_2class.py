model = dict(
    type='DeeplabV3Plus',
    encoder=dict(
        type='DeeplabV3PlusEncoder',
        backbone=dict(type='ResNet', name='resnet34', weights='imagenet'),
        encoder_output_stride=16),
    decoder=dict(
        type='DeeplabV3PlusDecoder',
        encoder_channels=(256, 64),
        decoder_channels=[256, 48],
        output_stride=16,
        final_channels=2))

data = dict(
    class_names=['Lesions'],
    samples_per_gpu=20,
    workers_per_gpu=4,
    train=dict(
        type='BaseDataset',
        data_root='./data/train',
        pipeline=[
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
        random_sampling=False,
        width=512,
        height=512,
        stride=512),
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
