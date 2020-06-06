import os

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
                 test_mode=False):
        super().__init__()
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.classes = classes
        self.img_paths, self.ann_paths = self._load_data(self.data_root)
        self._set_group_flag()

    def _load_data(self, data_root):
        names = os.listdir(os.path.join(data_root, 'images'))
        img_paths = [os.path.join(data_root, 'images', name) for name in names]
        ann_paths = [
            os.path.join(data_root, 'annotations', name) for name in names
        ]
        return img_paths, ann_paths

    def _get_data_info(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        input_dict = dict(img_path=img_path, ann_path=ann_path)
        return input_dict

    def _prepare_train_data(self, idx):
        input_dict = self._get_data_info(idx)
        example = self.pipeline(input_dict)
        return example

    def _prepare_test_data(self, idx):
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
        if self.test_mode:
            sample = self._prepare_test_data(idx)
        else:
            sample = self._prepare_train_data(idx)

        return sample

    def __len__(self):
        return len(self.img_paths)
