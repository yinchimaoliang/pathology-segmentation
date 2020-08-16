import os
from math import ceil

import cv2 as cv
import mmcv
import numpy as np


class Crop():

    def __init__(self,
                 data_root,
                 target_root,
                 width=512,
                 height=512,
                 stride=512):
        self.data_root = data_root
        self.target_root = target_root
        self.width = width
        self.height = height
        self.stride = stride

    def _crop(self):
        self.img_dict = {}
        self.ann_dict = {}
        # name, pos of the input
        self.infos = []
        for i, img_path in enumerate(self.img_paths):
            name = os.path.split(img_path)[-1].split('.')[0]
            img = cv.imread(img_path)
            ann = cv.imread(self.ann_paths[i], 0)
            if np.sum(ann > 3) > self.height * self.width / 100:
                continue
            else:
                ann[ann > 3] = 0
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
                    cv.imwrite(
                        os.path.join(self.target_root, 'images',
                                     f'{name}_{up}_{left}.png'),
                        img[up:up + self.height, left:left + self.width, :])
                    cv.imwrite(
                        os.path.join(self.target_root, 'annotations',
                                     f'{name}_{up}_{left}.png'),
                        img[up:up + self.height, left:left + self.width])

    def _get_infos(self):
        names = os.listdir(os.path.join(self.data_root, 'images'))
        self.img_paths = [
            os.path.join(self.data_root, 'images', name) for name in names
        ]
        self.ann_paths = [
            os.path.join(self.data_root, 'annotations', name) for name in names
        ]
        mmcv.mkdir_or_exist(os.path.join(self.target_root, 'images'))
        mmcv.mkdir_or_exist(os.path.join(self.target_root, 'annotations'))

    def main_func(self):
        self._get_infos()
        self._crop()


if __name__ == '__main__':
    crop = Crop('./data/train', './data/cropped/train')
    crop.main_func()
