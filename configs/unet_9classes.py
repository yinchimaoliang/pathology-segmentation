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
    train=dict(
        type='BaseDataset',
        data_root='./tests/data',
        pipeline=[
            dict(type='Loading'),
            dict(
                type='RandomSampling', prob_global=.5,
                target_shape=(512, 512)),
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
        test_mode=False),
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
        test_mode=True))

train = dict(
    loss=dict(type='BCEDiceLoss', ),
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    scheduler=dict(step_size=30, gamma=0.1))

valid = dict(evals=['Dsc', 'Iou'])
