import argparse
import os
import shutil

import mmcv
import numpy as np


def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--root_path',
        type=str,
        default='/home1/yinhaoli/data/gastric_mucosa/second_batch',
        help='specify the root path')
    parser.add_argument(
        '--target_path',
        type=str,
        default='/home1/yinhaoli/data/gastric_mucosa',
        help='specify the target path')
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='specify the target path')
    parser.add_argument(
        '--valid_ratio',
        type=float,
        default=0.3,
        help='specify the target path')
    args = parser.parse_args()
    return args


class DivideDataset():

    def __init__(self):
        args = parge_config()
        self.root_path = args.root_path
        self.target_path = args.target_path
        self.train_ratio = args.train_ratio
        self.valid_ratio = args.valid_ratio
        self.names = os.listdir(os.path.join(args.root_path, 'images'))

    def divide(self):
        np.random.shuffle(self.names)
        train_names = self.names[:int(self.train_ratio * len(self.names))]
        valid_names = self.names[int(self.train_ratio *
                                     len(self.names)):int(self.train_ratio *
                                                          len(self.names)) +
                                 int(self.valid_ratio * len(self.names))]
        test_names = self.names[int(self.train_ratio * len(self.names)) +
                                int(self.valid_ratio * len(self.names)):]
        train_path = os.path.join(self.target_path, 'train')
        valid_path = os.path.join(self.target_path, 'valid')
        test_path = os.path.join(self.target_path, 'test')
        mmcv.mkdir_or_exist(os.path.join(train_path, 'images'))
        mmcv.mkdir_or_exist(os.path.join(train_path, 'annotations'))
        mmcv.mkdir_or_exist(os.path.join(valid_path, 'images'))
        mmcv.mkdir_or_exist(os.path.join(valid_path, 'annotations'))
        mmcv.mkdir_or_exist(os.path.join(test_path, 'images'))
        mmcv.mkdir_or_exist(os.path.join(test_path, 'annotations'))
        for name in train_names:
            print(name)
            ann_name = name.split('.')[0] + '_mask.png'
            shutil.copyfile(
                os.path.join(self.root_path, 'images', name),
                os.path.join(self.target_path, 'train', 'images', name))
            shutil.copyfile(
                os.path.join(self.root_path, 'annotations_manual', ann_name),
                os.path.join(self.target_path, 'train', 'annotations',
                             ann_name))
        for name in valid_names:
            print(name)
            ann_name = name.split('.')[0] + '_mask.png'
            shutil.copyfile(
                os.path.join(self.root_path, 'images', name),
                os.path.join(self.target_path, 'valid', 'images', name))
            shutil.copyfile(
                os.path.join(self.root_path, 'annotations_manual', ann_name),
                os.path.join(self.target_path, 'valid', 'annotations',
                             ann_name))
        for name in test_names:
            print(name)
            ann_name = name.split('.')[0] + '_mask.png'
            shutil.copyfile(
                os.path.join(self.root_path, 'images', name),
                os.path.join(self.target_path, 'test', 'images', name))
            shutil.copyfile(
                os.path.join(self.root_path, 'annotations_manual', ann_name),
                os.path.join(self.target_path, 'test', 'annotations',
                             ann_name))


if __name__ == '__main__':
    divide_dataset = DivideDataset()
    divide_dataset.divide()
