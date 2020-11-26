import numpy as np

from .builder import EVALS


@EVALS.register_module()
class Dsc():

    def __init__(self, class_num):
        self.name = 'dsc'
        self.dscs = []
        self.dsc_per_class = [[] for _ in range(class_num - 1)]
        self.class_num = class_num
        self.num = 0

    def step(self, pr, gt):
        """

        :param pr: [h, w, c]
        :param gt: [h, w, c]
        :return:
        """
        epsilon = 1e-6
        inter = np.sum(np.bitwise_and(pr, gt), axis=(0, 2, 3))
        union = np.sum(pr, axis=(0, 2, 3)) + np.sum(gt, axis=(0, 2, 3))
        dsc = ((2. * inter + epsilon) / (union + epsilon))[1:]
        # print(iou)
        for i in range(1, self.class_num):
            if union[i] > 0:
                self.dsc_per_class[i - 1].append(dsc[i - 1])
        self.dscs.append(dsc)

        return np.mean(dsc, axis=0)

    def get_result(self):
        return np.mean(
            self.dscs, axis=0), [np.mean(dsc) for dsc in self.dsc_per_class]
