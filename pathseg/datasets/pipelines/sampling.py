import numpy as np

from pathseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RandomSampling():

    # TODO: add OTSU
    # TODO: add no target zone situation
    def __init__(self, prob_global, target_shape, filter_classes=[0]):

        self.prob_global = prob_global
        self.target_shape = target_shape
        self.filter_classes = filter_classes

    def __call__(self, results):
        img = results['image']
        ann = results['annotation']

        assert img.shape[:2] == ann.shape
        assert ann.shape[0] > self.target_shape[0] and ann.shape[
            1] > self.target_shape[1]

        if np.random.random() < self.prob_global:
            x = np.random.randint(0, img.shape[0] - self.target_shape[0])
            y = np.random.randint(0, img.shape[1] - self.target_shape[1])

        else:
            candidate = np.ones_like(ann)
            for filter_class in self.filter_classes:
                if np.sum(np.bitwise_and(candidate, ann != filter_class)) > 0:
                    candidate = np.bitwise_and(candidate, ann != filter_class)
            targets = np.transpose(np.array(np.where(candidate > 0)), (1, 0))
            target = targets[np.random.randint(targets.shape[0])]
            x = np.random.randint(
                max(0, target[0] - self.target_shape[0]),
                min(ann.shape[0] - self.target_shape[0], target[0] + 1))
            y = np.random.randint(
                max(0, target[1] - self.target_shape[1]),
                min(ann.shape[1] - self.target_shape[1], target[1] + 1))

        img = img[x:x + self.target_shape[0], y:y + self.target_shape[1], :]
        ann = ann[x:x + self.target_shape[0], y:y + self.target_shape[1]]
        results['image'] = img
        results['annotation'] = ann
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(prob_global={})'.format(self.prob_global)
        repr_str += '(target_shape={})'.format(self.target_shape)
        return repr_str
