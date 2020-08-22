import numpy as np

from .builder import EVALS


@EVALS.register_module()
class Dsc():

    def __init__(self, class_num):
        self.name = 'dsc'
        self.dscs = [[] for _ in range(class_num - 1)]
        self.class_num = class_num
        self.num = 0

    def step(self, pr, gt):
        """

        :param pr: [h, w, c]
        :param gt: [h, w, c]
        :return:
        """
        pr = pr.transpose(0, 2, 3, 1)
        gt = gt.transpose(0, 2, 3, 1)
        epsilon = 1e-6
        inter = np.sum(np.bitwise_and(pr, gt), axis=(0, 1, 2))
        union = np.sum(pr, axis=(0, 1, 2)) + np.sum(gt, axis=(0, 1, 2))
        dscs = ((2. * inter + epsilon) / (union + epsilon))[1:]
        for i, dsc in enumerate(dscs):
            if union[i + 1] > 0:
                self.dscs[i].append(dsc)

        return np.mean(dscs, axis=0)

    def get_result(self):
        return [np.mean(dsc) for dsc in self.dscs]
