import argparse
import os

import cv2 as cv
import mmcv
import numpy as np
import torch
import tqdm
from mmcv import Config
from tensorboardX import SummaryWriter

from pathseg.core.evals import build_eval
from pathseg.core.optimizers import build_optimizer
from pathseg.datasets import build_dataloader, build_dataset
from pathseg.models import build_loss, build_segmenter


def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='configs/unet_9classes.py',
        help='specify the config for training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        required=False,
        help='batch size for training')
    parser.add_argument(
        '--epochs',
        type=int,
        default=180,
        required=False,
        help='Number of epochs to train for')
    parser.add_argument(
        '--valid_per_iter',
        type=int,
        default=1,
        required=False,
        help='Number of Training epochs between valid')
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='number of workers for dataloader')
    parser.add_argument(
        '--extra_tag',
        type=str,
        default='default',
        help='extra tag for this experiment')
    args = parser.parse_args()
    return args


class Train():

    def __init__(self):
        self.args = parge_config()
        self.cfg = Config.fromfile(self.args.cfg_file)
        self.class_num = len(self.cfg.data.class_names) + 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = os.path.join('work_dirs', self.args.extra_tag)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.log_path = os.path.join(self.output_dir, 'logs')
        self.train_log_path = os.path.join(self.log_path, 'train')
        self.valid_log_path = os.path.join(self.log_path, 'valid')
        mmcv.mkdir_or_exist(self.log_path)
        mmcv.mkdir_or_exist(self.train_log_path)
        mmcv.mkdir_or_exist(self.valid_log_path)
        self.train_tb_log = SummaryWriter(self.train_log_path)
        self.valid_tb_log = SummaryWriter(self.valid_log_path)

    def train_one_epoch(self, tbar):

        total_it_each_epoch = len(self.train_data_loader)
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            leave=True,
            desc='train',
            dynamic_ncols=True)
        for ind, ret_dict in enumerate(self.train_data_loader):
            self.accumulated_iter += 1
            self.optim.zero_grad()
            images = ret_dict['image'].to(self.device)
            annotation = ret_dict['annotation'].to(self.device)
            outputs = self.segmenter(images)
            loss = self.criterion(outputs, annotation)
            pbar.update()
            pbar.set_postfix(dict(loss=loss.item()))
            tbar.set_postfix(dict(total_it=self.accumulated_iter))
            tbar.refresh()
            loss.backward()
            self.optim.step()
            self.train_tb_log.add_scalar('loss', loss.item(),
                                         self.accumulated_iter)
        pbar.close()

    def save_ckpt(self, epoch):
        optim_state = self.optim.state_dict()
        model_state = self.segmenter.state_dict()
        ckpt_state = dict(
            epoch=epoch, model_state=model_state, optim_state=optim_state)
        ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        ckpt_name = os.path.join(ckpt_dir, ('checkout_epoch_%d.pth' % epoch))
        torch.save(ckpt_state, ckpt_name)

    def evaluation(self, names, epoch, class_names):
        print(len(names))
        eval_dict = {}
        evals = [
            build_eval(dict(type=eval_name, class_num=self.class_num))
            for eval_name in self.cfg.valid.evals
        ]
        for eval in evals:
            for name in names:
                eval.step(
                    np.array([self.name_mask[name]]).transpose(0, 3, 1, 2),
                    np.array([self.name_anno[name]]).transpose(0, 3, 1, 2))
        for eval in evals:
            for i, class_name in enumerate(class_names):
                print(f'{eval.name}_{class_name}', eval.get_result()[i])
                # TODO: add name of each class.
                self.valid_tb_log.add_scalar(f'{eval.name}_{class_name}',
                                             eval.get_result()[i], epoch)
            print(f'm_{eval.name}', np.mean(eval.get_result()))
            self.valid_tb_log.add_scalar(f'm_{eval.name}',
                                         np.mean(eval.get_result()), epoch)
        for key, val in eval_dict.items():
            self.tb_log.add_scalar(key, val, epoch)

    def valid_one_epoch(self, epoch, class_names):
        self.name_mask = {}
        self.name_anno = {}
        names = os.listdir(
            os.path.join(self.cfg.data.valid.data_root, 'images'))
        for img_name in names:
            img = cv.imread(
                os.path.join(self.cfg.data.valid.data_root, 'images',
                             img_name), 0)
            self.name_mask[img_name] = np.zeros(
                (img.shape[0], img.shape[1], self.class_num))
            self.name_anno[img_name] = np.zeros(
                (img.shape[0], img.shape[1], self.class_num))
        total_it_each_epoch = len(self.valid_data_loader)
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            leave=True,
            desc='valid',
            dynamic_ncols=True)
        evals = [
            build_eval(dict(type=eval_name, class_num=self.class_num))
            for eval_name in self.cfg.valid.evals
        ]
        disp_dict = dict()
        for ind, ret_dict in enumerate(self.valid_data_loader):
            images = ret_dict['image'].to(self.device)
            annotations = ret_dict['annotation'].to(self.device)
            outputs = self.segmenter.predict(images)
            loss = self.criterion(outputs, annotations)
            disp_dict['loss'] = loss.item()
            annotations = annotations.cpu().numpy()
            outputs = outputs.data.cpu().numpy()
            for eval in evals:
                disp_dict[eval.name] = np.mean(eval.step(outputs, annotations))
            for i, output in enumerate(outputs):
                info = ret_dict['info']
                name = info[0][i]
                up = info[1][i].numpy()
                left = info[2][i].numpy()
                self.name_mask[name][
                    up:up + self.cfg.data.height, left:left +
                    self.cfg.data.width, :] += outputs[i].transpose(1, 2, 0)
                self.name_anno[name][
                    up:up + self.cfg.data.height, left:left +
                    self.cfg.data.width, :] = annotations[i].transpose(
                        1, 2, 0)

            pbar.update()
            pbar.set_postfix(disp_dict)

        self.evaluation(names, epoch, class_names)
        print('\n')

        pbar.close()
        self.save_ckpt(epoch)

    def main_func(self):
        self.segmenter = build_segmenter(self.cfg.model)
        self.segmenter.to(self.device)
        self.train_dataset = build_dataset(self.cfg.data.train)
        self.valid_dataset = build_dataset(self.cfg.data.valid)
        print('Train dataset : %d' % len(self.train_dataset))
        self.train_data_loader = build_dataloader(self.train_dataset,
                                                  self.args.batch_size,
                                                  self.args.workers)
        self.valid_data_loader = build_dataloader(self.valid_dataset,
                                                  self.args.batch_size,
                                                  self.args.workers)
        self.max_dsc = 0
        self.criterion = build_loss(self.cfg.train.loss)
        self.optim = build_optimizer(self.segmenter, self.cfg.train.optimizer)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=self.cfg.train.scheduler.step_size,
            gamma=self.cfg.train.scheduler.gamma)
        self.accumulated_iter = 0
        with tqdm.trange(0, self.args.epochs, desc='epochs') as tbar:
            for cur_epoch in tbar:
                self.train_one_epoch(tbar)

                if (cur_epoch + 1) % self.args.valid_per_iter == 0:
                    self.valid_one_epoch(cur_epoch, self.cfg.data.class_names)

                if cur_epoch < 59:
                    self.scheduler.step()
            print(self.cfg)


if __name__ == '__main__':
    train = Train()
    train.main_func()
