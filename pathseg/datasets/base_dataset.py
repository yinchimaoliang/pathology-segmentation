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
                 height=512,
                 repeat=1,
                 balance_class=False,
                 use_path=False,
                 drop_prob=0.95):
        super().__init__()
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.random_sampling = random_sampling
        self.classes = classes
        self.balance_class = balance_class
        self.use_path = use_path
        self.drop_prob = drop_prob
        self.img_paths, self.ann_paths = self._load_data(self.data_root)
        self.stride = stride
        self.width = width
        self.height = height
        self.repeat = repeat
        self._get_info()
        self._set_group_flag()
        if self.random_sampling or self.use_path:
            self.length = len(self.img_paths)
        else:
            self.length = len(self.infos)

    def _get_info(self):
        self.img_dict = {}
        self.ann_dict = {}
        # name, pos of the input
        self.infos = []
        self.imgs = []
        self.anns = []
        if self.balance_class:
            classes_img_paths = [[] for _ in range(len(self.classes))]
            classes_ann_paths = [[] for _ in range(len(self.classes))]
        for i, img_path in enumerate(self.img_paths):
            name = os.path.split(img_path)[-1]
            if self.use_path:
                if self.balance_class:
                    ann = cv.imread(self.ann_paths[i], 0)
                    if np.sum(ann == 0) == ann.shape[0] * ann.shape[1]:
                        if np.random.random() < self.drop_prob:
                            continue
                        else:
                            self.imgs.append(img_path)
                            self.anns.append(self.ann_paths[i])
                    for j in range(len(self.classes)):
                        if np.sum(ann == j) > 0:
                            classes_img_paths[j].append(img_path)
                            classes_ann_paths[j].append(self.ann_paths[i])

                else:
                    self.imgs.append(img_path)
                    self.anns.append(self.ann_paths[i])
            else:
                img = cv.imread(img_path)
                ann = cv.imread(self.ann_paths[i], 0)
                self.img_dict[name] = img
                self.ann_dict[name] = ann
                if not self.random_sampling:
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

        if self.balance_class:
            max_num = np.max(
                [len(class_name) for class_name in classes_img_paths])
            for i in range(len(self.classes)):
                class_img_paths = np.array(classes_img_paths[i])
                class_ann_paths = np.array(classes_ann_paths[i])
                if len(classes_img_paths[i]) > 0:
                    if len(classes_img_paths[i]) >= int(max_num):
                        choices = np.random.choice(
                            len(classes_img_paths[i]),
                            int(max_num),
                            replace=False)
                    else:
                        choices = np.random.choice(
                            len(classes_img_paths[i]), int(max_num))
                    self.imgs.extend(class_img_paths[choices])
                    self.anns.extend(class_ann_paths[choices])

    def _load_data(self, data_root):
        names = os.listdir(os.path.join(data_root, 'images'))
        img_paths = [os.path.join(data_root, 'images', name) for name in names]
        ann_paths = [
            os.path.join(data_root, 'annotations', name) for name in names
        ]
        return img_paths, ann_paths

    def _get_data_info(self, idx):
        if self.use_path:
            img = self.imgs[idx]
            ann = self.anns[idx]
            input_dict = dict(img_path=img, ann_path=ann)
        elif not self.random_sampling:
            info = self.infos[idx]
            name, up, left = info
            img = self.img_dict[name][up:up + self.height,
                                      left:left + self.width, :]
            ann = self.ann_dict[name][up:up + self.height,
                                      left:left + self.width]
            input_dict = dict(image=img, annotation=ann, info=info)
        else:
            img = list(self.img_dict.values())[idx]
            ann = list(self.ann_dict.values())[idx]

            input_dict = dict(image=img, annotation=ann)
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
        sample = self._prepare_data(idx % self.length)

        return sample

    def __len__(self):
        if self.random_sampling:
            return len(self.img_paths) * self.repeat
        elif self.use_path:
            return len(self.imgs)
        else:
            return len(self.infos)
