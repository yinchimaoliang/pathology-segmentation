import argparse
import os

import cv2 as cv
import mmcv
import numpy as np
import torch
import tqdm
from mmcv import Config

from pathseg.datasets import build_dataloader, build_dataset
from pathseg.models import build_segmentor


def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='./configs/unet_9classes.py',
        help='specify the config for training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        required=False,
        help='batch size for training')
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        required=False,
        help='The path of the checkpoint file')
    parser.add_argument(
        '--valid_per_iter',
        type=int,
        default=10,
        required=False,
        help='Number of Training epochs between valid')
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for dataloader')
    parser.add_argument(
        '--extra_tag',
        type=str,
        default='default',
        help='extra tag for this experiment')
    args = parser.parse_args()
    return args


class Inference():

    def __init__(self):
        self.args = parge_config()
        self.cfg = Config.fromfile(self.args.cfg_file)
        self.class_num = len(self.cfg.data.class_names) + 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = os.path.join('work_dirs', self.args.extra_tag)
        self.segmenter = build_segmentor(self.cfg.model)
        self.segmenter.to(self.device)
        state_dict = torch.load(self.args.ckpt_path)['model_state']
        self.segmenter.load_state_dict(state_dict)
        self.inference_dataset = build_dataset(self.cfg.data.inference)
        self.inference_data_loader = build_dataloader(self.inference_dataset,
                                                      self.args.batch_size,
                                                      self.args.workers)

    def visualize(self):
        result_dir = os.path.join(self.output_dir, 'results')
        colors = self.cfg.inference.colors
        mmcv.mkdir_or_exist(result_dir)
        for img_name in self.names:
            img = cv.imread(
                os.path.join(self.cfg.data.inference.data_root, 'images',
                             img_name))
            color_mask = np.zeros_like(img)
            mask = np.argmax(self.name_mask[img_name], 2).astype(np.uint8)
            for i, color in enumerate(colors):
                pos = np.where(mask == i + 1)
                color_mask[pos] = color
            res = cv.addWeighted(img, (1 - self.cfg.inference.weight),
                                 color_mask, self.cfg.inference.weight, 0)

            cv.imwrite(os.path.join(result_dir, img_name), res)

    def inference(self):
        self.name_mask = {}
        self.name_anno = {}
        self.names = os.listdir(
            os.path.join(self.cfg.data.inference.data_root, 'images'))
        for img_name in self.names:
            img = cv.imread(
                os.path.join(self.cfg.data.inference.data_root, 'images',
                             img_name), 0)
            self.name_mask[img_name] = np.zeros(
                (img.shape[0], img.shape[1], self.class_num))
            self.name_anno[img_name] = np.zeros(
                (img.shape[0], img.shape[1], self.class_num))
        total_it_each_epoch = len(self.inference_data_loader)
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            leave=True,
            desc='inference',
            dynamic_ncols=True)
        disp_dict = dict()
        for ind, ret_dict in enumerate(self.inference_data_loader):
            images = ret_dict['image'].to(self.device)
            annotations = ret_dict['annotation'].to(self.device)
            outputs = self.segmenter.predict(images)
            annotations = annotations.cpu().numpy()
            outputs = outputs.data.cpu().numpy()
            info = ret_dict['info']
            for i, output in enumerate(outputs):
                name = info[0][i]
                up = info[1][i].numpy()
                left = info[2][i].numpy()
                self.name_mask[name][
                    up:up + self.cfg.data.inference.height, left:left +
                    self.cfg.data.inference.width, :] += outputs[i].transpose(
                        1, 2, 0)
                self.name_anno[name][up:up + self.cfg.data.inference.height,
                                     left:left + self.cfg.data.inference.
                                     width, :] = annotations[i].transpose(
                                         1, 2, 0)

            pbar.update()
            pbar.set_postfix(disp_dict)
        print('\n')

        pbar.close()
        self.visualize()


if __name__ == '__main__':
    inference = Inference()
    inference.inference()
