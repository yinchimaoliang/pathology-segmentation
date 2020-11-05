import argparse
import os
import time

import cv2 as cv
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist
from tensorboardX import SummaryWriter

from pathseg.core.evals import build_eval
from pathseg.core.optimizers import build_optimizer
from pathseg.datasets import build_dataloader, build_dataset
from pathseg.models import build_loss, build_segmenter
from pathseg.utils import collect_env, get_root_logger


def parge_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument(
        '--cfg_file',
        type=str,
        default='configs/unet_resnet18_2classes.py',
        help='specify the config for training')
    parser.add_argument(
        '--epochs',
        type=int,
        default=180,
        required=False,
        help='Number of epochs to train for')
    parser.add_argument(
        '--valid_per_iter',
        type=int,
        default=10,
        required=False,
        help='Number of Training epochs between valid')
    parser.add_argument(
        '--extra_tag',
        type=str,
        default='default',
        help='extra tag for this experiment')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


class Train():

    def __init__(self):
        self.args = parge_config()
        self.cfg = Config.fromfile(self.args.cfg_file)
        if self.args.gpu_ids is None:
            self.args.gpu_ids = range(1) if self.args.gpus is None else range(
                self.args.gpus)
        self.class_num = len(self.cfg.data.class_names) + 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = os.path.join('work_dirs', self.args.extra_tag)
        mmcv.mkdir_or_exist(self.output_dir)
        self.tb_log_path = os.path.join(self.output_dir, 'logs')
        self.train_tb_log_path = os.path.join(self.tb_log_path, 'train')
        self.valid_tb_log_path = os.path.join(self.tb_log_path, 'valid')
        mmcv.mkdir_or_exist(self.tb_log_path)
        mmcv.mkdir_or_exist(self.train_tb_log_path)
        mmcv.mkdir_or_exist(self.valid_tb_log_path)
        self.train_tb_log = SummaryWriter(self.train_tb_log_path)
        self.valid_tb_log = SummaryWriter(self.valid_tb_log_path)
        self.segmenter = build_segmenter(self.cfg.model)
        self.train_dataset = build_dataset(self.cfg.data.train)
        self.valid_dataset = build_dataset(self.cfg.data.valid)
        print('Train dataset : %d' % len(self.train_dataset))
        self.train_data_loader = build_dataloader(
            self.train_dataset, self.cfg.data.samples_per_gpu,
            self.cfg.data.workers_per_gpu, len(self.args.gpu_ids))
        self.valid_data_loader = build_dataloader(
            self.valid_dataset, self.cfg.data.samples_per_gpu,
            self.cfg.data.workers_per_gpu)
        self.max_dsc = 0
        self.criterion = build_loss(self.cfg.train.loss)
        self.optim = build_optimizer(self.segmenter, self.cfg.train.optimizer)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=self.cfg.train.scheduler.step_size,
            gamma=self.cfg.train.scheduler.gamma)
        self.accumulated_iter = 0
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.output_dir, f'{timestamp}.log')
        self.logger = get_root_logger(
            log_file=log_file, log_level=self.cfg.log_level)
        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher == 'none':
            self.distributed = False
        else:
            self.distributed = True
            init_dist(self.args.launcher, **self.cfg.dist_params)

        # put model on gpus
        if self.distributed:
            find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                  False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            self.segmenter = MMDistributedDataParallel(
                self.segmenter.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            self.segmenter = MMDataParallel(
                self.segmenter.cuda(self.args.gpu_ids[0]),
                device_ids=self.args.gpu_ids)

    def train_one_epoch(self, tbar):

        total_it_each_epoch = len(self.train_data_loader)
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            leave=True,
            desc='train',
            dynamic_ncols=True)
        losses = []
        for ind, ret_dict in enumerate(self.train_data_loader):
            self.accumulated_iter += 1
            self.optim.zero_grad()
            images = ret_dict['image'].to(self.device)
            annotation = ret_dict['annotation'].to(self.device)
            outputs = self.segmenter(images)
            loss = self.criterion(outputs, annotation)
            losses.append(loss.data.cpu().numpy())
            pbar.update()
            pbar.set_postfix(dict(loss=loss.item()))
            tbar.set_postfix(dict(total_it=self.accumulated_iter))
            tbar.refresh()
            loss.backward()
            self.optim.step()
            self.train_tb_log.add_scalar('loss', loss.item(),
                                         self.accumulated_iter)
        pbar.close()
        return sum(losses) / len(losses)

    def save_ckpt(self, epoch):
        optim_state = self.optim.state_dict()
        model_state = self.segmenter.module.state_dict()
        ckpt_state = dict(
            epoch=epoch, model_state=model_state, optim_state=optim_state)
        ckpt_dir = os.path.join(self.output_dir, 'ckpt')
        mmcv.mkdir_or_exist(ckpt_dir)
        ckpt_name = os.path.join(ckpt_dir, ('checkout_epoch_%d.pth' % epoch))
        torch.save(ckpt_state, ckpt_name)

    def evaluation(self, names, epoch, class_names):
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
        self.segmenter.eval()
        self.name_mask = {}
        self.name_anno = {}
        names = os.listdir(
            os.path.join(self.cfg.data.valid.data_root, 'images'))
        for img_name in names:
            img = cv.imread(
                os.path.join(self.cfg.data.valid.data_root, 'images',
                             img_name), 0)
            self.name_mask[img_name] = np.zeros(
                (img.shape[0], img.shape[1], self.class_num), dtype=np.bool)
            self.name_anno[img_name] = np.zeros(
                (img.shape[0], img.shape[1], self.class_num), dtype=np.bool)
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
            with torch.no_grad():
                outputs = self.segmenter(images)
            loss = self.criterion(outputs, annotations)
            disp_dict['loss'] = loss.item()
            annotations = annotations.cpu().numpy()
            outputs = F.interpolate(outputs[0],
                                    (annotations.shape[2],
                                     annotations.shape[3])).data.cpu().numpy()
            outputs = np.eye(
                self.class_num, dtype=np.bool)[np.argmax(
                    outputs.transpose(0, 2, 3, 1), axis=3)]
            annotations = annotations.transpose(0, 2, 3, 1)
            if 'info' in ret_dict.keys():
                info = ret_dict['info']
                for eval in evals:
                    disp_dict[eval.name] = np.mean(
                        eval.step(outputs, annotations))
                for i, output in enumerate(outputs):
                    name = info[0][i]
                    up = info[1][i].numpy()
                    left = info[2][i].numpy()
                    self.name_mask[name][
                        up:up + self.cfg.data.valid.height,
                        left:left + self.cfg.data.valid.width, :] += outputs[i]
                    self.name_anno[name][
                        up:up + self.cfg.data.valid.height, left:left +
                        self.cfg.data.valid.width, :] = annotations[i]
            else:
                for eval in evals:
                    disp_dict[eval.name] = np.mean(
                        eval.step(outputs, annotations))
                for i, output in enumerate(outputs):
                    name = ret_dict['name'][i]
                    self.name_mask[name] = outputs[i]
                    self.name_anno[name] = annotations[i]
            pbar.update()
            pbar.set_postfix(disp_dict)

        self.evaluation(names, epoch, class_names)
        print('\n')

        pbar.close()
        self.save_ckpt(epoch + 1)
        self.segmenter.train()

    def main_func(self):
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                         dash_line)
        self.logger.info(self.segmenter)
        with tqdm.trange(0, self.args.epochs, desc='epochs') as tbar:
            for cur_epoch in tbar:
                loss = self.train_one_epoch(tbar)
                cur_lr = self.optim.state_dict()['param_groups'][0]['lr']
                self.logger.info(
                    f'epoch: {cur_epoch}, learning rate: {cur_lr}, '
                    f'loss: {loss}')

                if (cur_epoch + 1) % self.args.valid_per_iter == 0:
                    self.valid_one_epoch(cur_epoch, self.cfg.data.class_names)

                if cur_epoch < 100:
                    self.scheduler.step()


if __name__ == '__main__':
    train = Train()
    train.main_func()
