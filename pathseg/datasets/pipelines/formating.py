import numpy as np
import torch
import torchvision.transforms as tfs

from pathseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class Formating():

    def __init__(self, mean, std, num_classes):
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

    def __call__(self, results):
        img = results['image']
        ann = results['annotation']
        im_tfs = tfs.Compose([
            tfs.ToTensor(),  # [0-255]--->[0-1]
            tfs.Normalize(self.mean, self.std)
        ])
        img = im_tfs(img)
        ann = np.eye(self.num_classes, dtype=np.bool)[ann]
        ann = ann.transpose(2, 0, 1)
        ann = torch.from_numpy(ann)

        results['image'] = img
        results['annotation'] = ann

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={})'.format(self.mean)
        repr_str += '(std={})'.format(self.std)
        repr_str += '(num_classes={})'.format(self.num_classes)
        return repr_str
