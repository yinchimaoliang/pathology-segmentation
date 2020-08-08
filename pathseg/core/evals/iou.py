import numpy as np

from .builder import EVALS


@EVALS.register_module()
class Iou():

    def __init__(self, class_num):
        self.name = 'iou'
        self.ious = []
        self.class_num = class_num
        self.num = 0

    def step(self, pr, gt):

        pr = pr.transpose(0, 2, 3, 1)
        gt = gt.transpose(0, 2, 3, 1)
        self.num += 1
        epsilon = 1e-6
        inter = np.sum(np.bitwise_and(pr, gt), axis=(0, 1, 2))
        union = np.sum(np.bitwise_or(pr, gt), axis=(0, 1, 2))
        iou = ((inter + epsilon) / (union + epsilon))[1:]
        # print(iou)
        self.ious.append(iou)

        return np.mean(iou, axis=0)

    def get_result(self):
        return np.mean(self.ious, axis=0)
