import numpy as np

from .builder import EVALS


@EVALS.register_module()
class Dsc():

    def __init__(self, class_num):
        self.name = 'dsc'
        self.dscs = []
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
        # print(pr.shape)
        # print(gt.shape)
        y_true = np.eye(self.class_num)[np.argmax(pr, axis=3)].astype(np.bool)
        # print('y_true : ', y_true.shape)
        y_pred = gt.astype(np.bool)
        inter = np.sum(np.bitwise_and(y_true, y_pred), axis=(0, 1, 2))
        union = np.sum(y_true, axis=(0, 1, 2)) + np.sum(y_pred, axis=(0, 1, 2))
        dsc = ((2. * inter + epsilon) / (union + epsilon))[1:]
        # print(iou)
        self.dscs.append(dsc)

        return np.mean(dsc, axis=0)

    def get_result(self):
        return np.mean(self.dscs, axis=0)
