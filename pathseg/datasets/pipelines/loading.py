import cv2 as cv

from pathseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Loading(object):

    def __init__(self, shape=None):
        self.shape = shape

    def __call__(self, results):
        img_path = results['img_path']
        ann_path = results['ann_path']

        img = cv.imread(img_path)
        ann = cv.imread(ann_path, 0)

        if self.shape is not None:
            if isinstance(self.shape, float):
                img = cv.resize(img, (int(img.shape[1] * self.shape),
                                      int(img.shape[0] * self.shape)))
                ann = cv.resize(ann, (int(ann.shape[1] * self.shape),
                                      int(ann.shape[0] * self.shape)))
            else:
                img = cv.resize(img, self.shape)
                ann = cv.resize(ann, self.shape)

        results['image'] = img
        results['annotation'] = ann
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(shape={})'.format(self.shape)
        return repr_str
