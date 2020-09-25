import numpy as np

from .builder import EVALS


@EVALS.register_module()
class MAE():

    def __init__(self, num_class):
        self.num_class = num_class
        self.maes = []
        self.num = 0
        self.name = 'MAE'

    def step(self, pr, gt):
        mae = np.abs(pr - gt)
        self.maes.append(mae)
        return np.mean(mae)

    def get_result(self):
        return np.mean(self.maes, axis=0)
