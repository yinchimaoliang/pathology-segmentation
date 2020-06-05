import cv2 as cv
import torch

from pathseg.datasets.pipelines import Formating


def test_formating():
    formating = Formating([0.5, 0.5, 0.5], [0.1, 0.1, 0.1], 9)

    img = cv.imread('./tests/data/images/test.png')
    ann = cv.imread('./tests/data/annotations/test.png', 0)
    results = dict(image=img, annotation=ann)
    results = formating(results)
    img = results['image']
    ann = results['annotation']

    assert isinstance(img, torch.Tensor)
    assert isinstance(ann, torch.Tensor)
    assert img.shape == torch.Size([3, 5736, 3538])
    assert ann.shape == torch.Size([9, 5736, 3538])
