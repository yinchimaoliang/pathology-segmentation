import cv2 as cv
from pathseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Loading(object):

    def __init__(self, ):
        pass

    def __call__(self, results):
        img_path = results['img_path']
        ann_path = results['ann_path']

        img = cv.imread(img_path)
        ann = cv.imread(ann_path, 0)

        results['image'] = img
        results['annotation'] = ann

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(shift_height={})'.format(self.shift_height)
        repr_str += '(mean_color={})'.format(self.color_mean)
        repr_str += '(load_dim={})'.format(self.load_dim)
        repr_str += '(use_dim={})'.format(self.use_dim)
        return repr_str
