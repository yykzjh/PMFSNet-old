import os
import torch
import numpy as np

import configs.config as config

from lib import utils
from lib.visualizations.TensorboardWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(self, model, criterion, optimizer, train_data_loader,
                 valid_data_loader=None, lr_scheduler=None, device=None):

        # 传入的参数
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.device = device

        # 创建训练执行目录和文件
        self.execute_dir = os.path.join(
            config.runs_dir,
            "{}_{}".format(config.model_name, config.dataset_name),
            utils.datestr()
        )
        self.tensorboard_dir = os.path.join(self.execute_dir, "board")
        self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
        self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
        utils.make_dirs(self.tensorboard_dir)
        utils.make_dirs(self.checkpoint_dir)

        # 训练时需要用到的参数
        self.best_dice = config.best_dice
        self.terminal_show_freq = config.terminal_show_freq
        self.per_epoch_total_step = len(self.train_data_loader)
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(self.tensorboard_dir, self.log_txt_path)


    def training(self):
        for epoch in range(config.start_epoch, config.end_epoch):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_dice = self.writer.data['val']['dsc'] / self.writer.data['val']['count']

            if self.checkpoint_dir is not None:
                self.model.save_checkpoint(self.checkpoint_dir, epoch, type="last", optimizer=self.optimizer)
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self.model.save_checkpoint(self.checkpoint_dir, epoch, type="best", best_metric=self.best_dice,
                                               optimizer=self.optimizer, metric_name="dice")

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')
        self.writer.close_writer()


    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()

            input_tensor, target = utils.prepare_input(input_tuple=input_tuple, device=self.device)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.per_epoch_total_step + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                self.writer.display_terminal(epoch, batch_idx, self.per_epoch_total_step, mode='train')

        self.writer.display_terminal(epoch, mode='train', summary=True)


    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = utils.prepare_input(input_tuple=input_tuple, device=self.device)
                input_tensor.requires_grad = False

                output = self.model(input_tensor)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.per_epoch_total_step + batch_idx)

        self.writer.display_terminal(epoch, mode='val', summary=True)
