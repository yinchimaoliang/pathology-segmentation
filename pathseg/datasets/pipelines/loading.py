import cv2 as cv
import numpy as np

from pathseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Loading(object):

    def __init__(self, shape, num_class):
        self.shape = shape
        self.num_class = num_class

    def __call__(self, results):
        img_path = results['img_path']
        ann_path = results['ann_path']

        img = cv.imread(img_path)
        ann_pixel = cv.imread(ann_path, 0)

        if self.shape is not None:
            img = cv.resize(img, self.shape)
            ann_pixel = cv.resize(ann_pixel, self.shape)

        ann = []
        for i in range(self.num_class):
            ann.append(
                np.sum(ann_pixel == i + 1) /
                (ann_pixel.shape[0] * ann_pixel.shape[1]))
        results['image'] = img
        results['annotation'] = np.array(ann)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(shape={})'.format(self.shape)
        return repr_str
