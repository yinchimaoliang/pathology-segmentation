import os
from math import ceil

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class BaseDataset(Dataset):

    def __init__(self,
                 data_root,
                 pipeline=None,
                 classes=None,
                 random_sampling=False,
                 stride=512,
                 width=512,
                 height=512):
        super().__init__()
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.random_sampling = random_sampling
        self.classes = classes
        self.img_paths, self.ann_paths = self._load_data(self.data_root)
        self.stride = stride
        self.width = width
        self.height = height
        if not self.random_sampling:
            self._get_info()
        self._set_group_flag()

    def _get_info(self):
        self.img_dict = {}
        self.ann_dict = {}
        # name, pos of the input
        self.infos = []
        for i, img_path in enumerate(self.img_paths):
            name = os.path.split(img_path)[-1]
            img = cv.imread(img_path)
            ann = cv.imread(self.ann_paths[i], 0)
            self.img_dict[name] = img
            self.ann_dict[name] = ann
            height = img.shape[0]
            width = img.shape[1]
            for i in range(int(ceil(height / self.stride))):
                for j in range(int(ceil(width / self.stride))):
                    if j * self.stride + self.width < width:
                        left = j * self.stride
                    else:
                        left = width - self.width
                    if i * self.stride + self.height < height:
                        up = i * self.stride
                    else:
                        up = height - self.height
                    self.infos.append([name, up, left])

    def _load_data(self, data_root):
        names = os.listdir(os.path.join(data_root, 'images'))
        img_paths = [os.path.join(data_root, 'images', name) for name in names]
        ann_paths = [
            os.path.join(data_root, 'annotations', name) for name in names
        ]
        return img_paths, ann_paths

    def _get_data_info(self, idx):
        if not self.random_sampling:
            info = self.infos[idx]
            name, up, left = info
            img = self.img_dict[name][up:up + self.height,
                                      left:left + self.width, :]
            ann = self.ann_dict[name][up:up + self.height,
                                      left:left + self.width]
            input_dict = dict(image=img, annotation=ann, info=info)
        else:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]

            input_dict = dict(img_path=img_path, ann_path=ann_path)
        return input_dict

    def _prepare_data(self, idx):
        input_dict = self._get_data_info(idx)
        example = self.pipeline(input_dict)
        return example

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        In 3D datasets, they are all the same, thus are all zeros

        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        sample = self._prepare_data(idx)

        return sample

    def __len__(self):
        if self.random_sampling:
            return len(self.img_paths)
        else:
            return len(self.infos)
