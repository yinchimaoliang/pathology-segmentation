from torch.utils.data.dataloader import DataLoader

from pathseg.datasets.builder import build_dataloader, build_dataset


def test_data_loader():
    pipelines = [
        dict(type='Loading'),
        dict(
            type='Flip',
            prob=1.,
            flip_ratio_horizontal=1.,
            flip_ratio_vertical=1.),
        dict(
            type='Flip',
            prob=1.,
            flip_ratio_horizontal=1.,
            flip_ratio_vertical=1.),
        dict(type='ShiftScaleRotate', prob=1.),
        dict(type='RandomRotate90', prob=1.),
        dict(type='RandomSampling', prob_global=0, target_shape=(512, 512)),
        dict(
            type='Formating',
            mean=[0.5, 0.5, 0.5],
            std=[0.1, 0.1, 0.1],
            num_classes=9)
    ]

    base_dataset = build_dataset(
        dict(
            type='BaseDataset',
            data_root='./tests/data',
            pipeline=pipelines,
            test_mode=False))

    data_loader = build_dataloader(
        dataset=base_dataset, samples_per_gpu=1, workers_per_gpu=0)

    assert isinstance(data_loader, DataLoader)
