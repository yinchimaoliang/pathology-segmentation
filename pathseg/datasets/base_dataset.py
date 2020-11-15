import os
from math import ceil

import mmcv
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
                 use_patch=True,
                 random_sampling=False,
                 horizontal_stride=512,
                 vertical_stride=512,
                 patch_width=512,
                 patch_height=512,
                 repeat=1):
        super().__init__()
        self.data_root = data_root
        self.pipeline = Compose(pipeline)
        self.use_patch = use_patch
        self.random_sampling = random_sampling
        self.classes = classes
        self.img_paths, self.ann_paths = self._load_data(self.data_root)
        self.horizontal_stride = horizontal_stride
        self.vertical_stride = vertical_stride
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.repeat = repeat
        if self.use_patch:
            self._get_info()
        self._set_group_flag()
        if self.use_patch:
            if self.random_sampling:
                self.length = len(self.img_paths)
            else:
                self.length = len(self.infos)
        else:
            self.length = len(self.img_paths)

    def _get_info(self):
        self.img_dict = {}
        self.ann_dict = {}
        # name, pos of the input
        self.infos = []
        for i, img_path in enumerate(self.img_paths):
            name = os.path.split(img_path)[-1]
            img = mmcv.imread(img_path)
            ann = mmcv.imread(self.ann_paths[i], 'grayscale')
            self.img_dict[name] = img
            self.ann_dict[name] = ann
            if not self.random_sampling:
                img_height = img.shape[0]
                img_width = img.shape[1]
                for i in range(int(ceil(img_height / self.vertical_stride))):
                    for j in range(
                            int(ceil(img_width / self.horizontal_stride))):
                        if j * self.horizontal_stride + self.patch_width \
                                < img_width:
                            left = j * self.horizontal_stride
                        else:
                            left = img_width - self.patch_width
                        if i * self.vertical_stride + self.patch_height \
                                < img_height:
                            up = i * self.vertical_stride
                        else:
                            up = img_height - self.patch_height
                        self.infos.append(
                            dict(
                                filename=name,
                                up=up,
                                left=left,
                                patch_height=self.patch_height,
                                patch_width=self.patch_width))
            else:
                self.infos.append(dict(filename=name))

    def _load_data(self, data_root):
        self.names = os.listdir(os.path.join(data_root, 'images'))
        img_paths = [
            os.path.join(data_root, 'images', name) for name in self.names
        ]
        ann_paths = [
            os.path.join(data_root, 'annotations',
                         name.split('.')[0] + '_mask.png')
            for name in self.names
        ]
        return img_paths, ann_paths

    def _get_data_info(self, idx):
        if self.use_patch:
            if not self.random_sampling:

                info = self.infos[idx]
                filename = info['filename']
                img = self.img_dict[filename]
                ann = self.ann_dict[filename]
                input_dict = dict(
                    img_prefix=os.path.join(self.data_root, 'images'),
                    img=img,
                    gt_semantic_seg=ann,
                    img_info=info,
                    seg_fields=['gt_semantic_seg'])
            else:
                info = self.infos[idx]
                filename = os.path.join(self.data_root, 'images',
                                        info['filename'])

                img = list(self.img_dict.values())[idx]
                ann = list(self.ann_dict.values())[idx]
                num_channels = 1 if len(img.shape) < 3 else img.shape[2]
                input_dict = dict(
                    img_prefix=os.path.join(self.data_root, 'images'),
                    filename=filename,
                    ori_filename=info['filename'],
                    ori_shape=img.shape,
                    pad_shape=img.shape,
                    scale_factor=1.0,
                    img_norm_cfg=dict(
                        mean=np.zeros(num_channels, dtype=np.float32),
                        std=np.ones(num_channels, dtype=np.float32),
                        to_rgb=False),
                    img=img,
                    gt_semantic_seg=ann,
                    img_info=info,
                    seg_fields=['gt_semantic_seg'])
        else:
            input_dict = dict(
                img_info=dict(filename=self.img_paths[idx]),
                ann_info=dict(seg_map=self.ann_paths[idx]),
                seg_fields=[])
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
        if self.use_patch:
            if self.random_sampling:
                return len(self.img_paths) * self.repeat
            else:
                return len(self.infos)
        else:
            return len(self.img_paths)
