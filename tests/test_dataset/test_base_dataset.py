import numpy as np
import torch

from pathseg.datasets.base_dataset import BaseDataset


def test_base_dataset():
    np.random.seed(0)
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
    base_dataset = BaseDataset(
        data_root='./tests/data', pipeline=pipelines, test_mode=False)
    sample = base_dataset[0]

    img = sample['image']
    ann = sample['annotation']

    assert isinstance(img, torch.Tensor)
    assert isinstance(ann, torch.Tensor)
    img_sample = img[:, 200:205, 200:205]
    expected_img_sample = torch.Tensor(
        [[[2.4902, 3.1569, 3.7843, 3.9804, 3.8627],
          [1.7059, 2.2157, 2.7255, 2.9216, 3.3529],
          [1.2353, 1.4314, 1.7451, 1.9412, 2.3725],
          [1.0392, 1.0784, 1.1961, 1.2745, 1.4706],
          [1.2745, 1.1569, 1.0784, 1.0784, 1.1176]],
         [[0.4902, 1.3922, 2.2157, 2.6863, 2.7255],
          [-0.5294, 0.1765, 0.8431, 1.2353, 1.7451],
          [-1.3137, -0.9216, -0.4510, -0.1373, 0.3333],
          [-1.7451, -1.5882, -1.3922, -1.2353, -0.9608],
          [-1.7843, -1.8235, -1.8627, -1.8235, -1.7451]],
         [[2.7255, 3.1176, 3.4706, 3.7451, 3.8627],
          [1.9804, 2.2941, 2.5686, 2.7255, 3.2353],
          [1.4706, 1.5490, 1.7059, 1.8235, 2.2941],
          [1.2353, 1.1569, 1.1961, 1.2353, 1.4314],
          [1.3922, 1.1961, 1.0784, 1.0784, 1.1569]]])
    assert torch.allclose(img_sample, expected_img_sample, 1e-3)
    assert img.shape == torch.Size([3, 512, 512])
    assert ann.shape == torch.Size([9, 512, 512])
