import cv2 as cv
import numpy as np
from pathseg.datasets.pipelines import Flip, RandomRotate90, ShiftScaleRotate


def test_flip():
    flip = Flip(1, 1, 1)

    img = np.array([[[1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
                   dtype=np.uint8)
    ann = np.array([[1, 0], [0, 0]], dtype=np.uint8)

    results = dict(image=img, annotation=ann)

    results = flip(results)

    img = results['image']
    ann = results['annotation']

    expected_img = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]],
                            dtype=np.uint8)
    expected_ann = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert np.all(img == expected_img)
    assert np.all(ann == expected_ann)


def test_shift_scale_rotate():
    np.random.seed(0)
    shift_scale_rotate = ShiftScaleRotate()

    img = cv.imread('./tests/data/images/test.png')
    ann = cv.imread('./tests/data/annotations/test.png', 0)

    results = dict(image=img, annotation=ann)

    results = shift_scale_rotate(results)

    img = results['image']
    ann = results['annotation']

    assert img.shape == (5736, 3538, 3)
    assert ann.shape == (5736, 3538)


def test_random_rotate90():
    np.random.seed(0)
    random_rotate90 = RandomRotate90(1.)
    img = cv.imread('./tests/data/images/test.png')
    ann = cv.imread('./tests/data/annotations/test.png', 0)

    results = dict(image=img, annotation=ann)

    results = random_rotate90(results)

    img = results['image']
    ann = results['annotation']

    assert img.shape == (5736, 3538, 3)
    assert ann.shape == (5736, 3538)
