import os
import os.path as osp
from functools import reduce
from math import ceil

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from pathseg.core import mean_iou
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class BaseDataset(Dataset):
    CLASSES = None

    PALETTE = None

    def __init__(self,
                 data_root,
                 pipeline=None,
                 use_patch=True,
                 random_sampling=False,
                 horizontal_stride=512,
                 vertical_stride=512,
                 patch_width=512,
                 patch_height=512,
                 repeat=1,
                 classes=None,
                 palette=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False):
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
        self.label_map = None
        self.repeat = repeat
        if self.use_patch:
            self._get_info()
        self._set_group_flag()
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        if self.use_patch:
            if self.random_sampling:
                self.length = len(self.img_paths)
            else:
                self.length = len(self.infos)
        else:
            self.length = len(self.img_paths)
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def _get_info(self):
        self.img_dict = {}
        self.ann_dict = {}
        # name, pos of the input
        self.infos = []
        for i, img_path in enumerate(self.img_paths):
            name = osp.split(img_path)[-1]
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
        self.names = os.listdir(osp.join(data_root, 'images'))
        img_paths = [
            osp.join(data_root, 'images', name) for name in self.names
        ]
        ann_paths = [
            osp.join(data_root, 'annotations',
                     name.split('.')[0] + '_mask.png') for name in self.names
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
                    img_prefix=osp.join(self.data_root, 'images'),
                    img=img,
                    gt_semantic_seg=ann,
                    img_info=info,
                    seg_fields=['gt_semantic_seg'])
            else:
                info = self.infos[idx]
                filename = osp.join(self.data_root, 'images', info['filename'])

                img = list(self.img_dict.values())[idx]
                ann = list(self.ann_dict.values())[idx]
                num_channels = 1 if len(img.shape) < 3 else img.shape[2]
                input_dict = dict(
                    img_prefix=osp.join(self.data_root, 'images'),
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

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for ann_path in self.ann_paths:
            gt_seg_map = mmcv.imread(
                ann_path, flag='grayscale', backend='pillow')
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        all_acc, acc, iou = mean_iou(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

        iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        summary_str += line_format.format('global', iou_str, acc_str,
                                          all_acc_str)
        print_log(summary_str, logger)

        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc

        return eval_results

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
