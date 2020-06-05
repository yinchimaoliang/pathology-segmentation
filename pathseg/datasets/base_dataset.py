import os

from torch.utils.data import Dataset

from pathseg.datasets import DATASETS
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
        self.img_paths, self.ann_paths = self.load_data(self.data_root)

    def load_data(self, data_root):
        names = os.listdir(os.path.join(data_root, 'images'))
        img_paths = [os.path.join(data_root, 'images', name) for name in names]
        ann_paths = [
            os.path.join(data_root, 'annotations', name) for name in names
        ]
        return img_paths, ann_paths

    def get_data_info(self, idx):
        img_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        input_dict = dict(img_path=img_path, ann_path=ann_path)
        return input_dict

    def prepare_train_data(self, idx):
        input_dict = self.get_data_info(idx)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, idx):
        input_dict = self.get_data_info(idx)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            sample = self.prepare_test_data(idx)
        else:
            sample = self.prepare_train_data(idx)

        return sample
