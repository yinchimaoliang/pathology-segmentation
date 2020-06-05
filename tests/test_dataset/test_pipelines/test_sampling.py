import cv2 as cv
import numpy as np

from pathseg.datasets.pipelines import RandomSampling


def test_random_sampling():

    np.random.seed(0)
    random_sampling = RandomSampling(1, (512, 512))

    img = cv.imread('./tests/data/images/test.png')
    ann = cv.imread('./tests/data/annotations/test.png', 0)
    results = dict(image=img, annotation=ann)
    random_sampling(results)

    img = results['image']
    ann = results['annotation']

    assert img.shape == (512, 512, 3)
    assert ann.shape == (512, 512)
