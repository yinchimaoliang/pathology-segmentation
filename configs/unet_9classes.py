model = dict(
    type='UNet',
    encoder=dict(
        type='UnetEncoder',
        backbone=dict(type='ResNet', name='resnet18', weights='imagenet'),
    ),
    decoder=dict(
        type='UnetDecoder',
        decoder_channels=(512, 256, 128, 64, 64),
        final_channels=9),
    activation='softmax')

data = dict(
    class_names=[1, 2, 3, 4, 5, 6, 7, 8],
    train=dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=[
            dict(
                type='Flip',
                prob=.5,
                flip_ratio_horizontal=.5,
                flip_ratio_vertical=.5),
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
                num_classes=9)
        ],
        random_sampling=False),
    valid=dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=9)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512),
    inference=dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=[
            dict(
                type='Formating',
                mean=[0.5, 0.5, 0.5],
                std=[0.1, 0.1, 0.1],
                num_classes=9)
        ],
        random_sampling=False,
        width=512,
        height=512,
        stride=512,
    ))

train = dict(
    loss=dict(type='BCEDiceLoss', ),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    scheduler=dict(step_size=30, gamma=0.1))

valid = dict(evals=['Dsc', 'Iou'])

inference = dict(
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [128, 128, 0], [0, 128, 128]],
    weight=0.2)
log_level = 'INFO'
