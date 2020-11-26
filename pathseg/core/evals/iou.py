import numpy as np

from .builder import EVALS


@EVALS.register_module()
class Iou():

    def __init__(self, class_num):
        self.name = 'iou'
        self.ious = []
        self.ious_per_class = [[] for _ in range(class_num - 1)]
        self.class_num = class_num
        self.num = 0

    def step(self, pr, gt):
        self.num += 1
        epsilon = 1e-6
        inter = np.sum(np.bitwise_and(pr, gt), axis=(0, 2, 3))
        union = np.sum(np.bitwise_or(pr, gt), axis=(0, 2, 3))
        iou = ((inter + epsilon) / (union + epsilon))[1:]
        # print(iou)
        for i in range(1, self.class_num):
            if union[i] > 0:
                self.ious_per_class[i - 1].append(iou[i - 1])
        self.ious.append(iou)

        return np.mean(iou, axis=0)

    def get_result(self):
        return np.mean(
            self.ious, axis=0), [np.mean(iou) for iou in self.ious_per_class]
