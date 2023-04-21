import os
import torch
import numpy as np

from lib import utils


class Trainer:
    """
    Trainer class
    """

    def __init__(self, opt, model, criterion, optimizer, train_loader,
                 valid_loader=None, lr_scheduler=None):

        # 传入的参数
        self.opt = opt
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data_loader = train_loader
        self.valid_data_loader = valid_loader
        self.lr_scheduler = lr_scheduler
        self.device = opt["device"]

        # 创建训练执行目录和文件
        self.execute_dir = os.path.join(opt["runs_dir"], utils.datestr())
        self.checkpoint_dir = os.path.join(self.execute_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.execute_dir, "board")
        self.log_txt_path = os.path.join(self.execute_dir, "log.txt")
        utils.make_dirs(self.checkpoint_dir)
        utils.make_dirs(self.tensorboard_dir)


        # 训练时需要用到的参数
        self.best_dice = opt["best_dice"]
        self.terminal_show_freq = opt["terminal_show_freq"]


    def training(self):
        for epoch in range(self.opt["start_epoch"], self.opt["end_epoch"]):
            # 当前epoch的训练阶段
            self.train_epoch(epoch)

            # 当前epoch的验证阶段
            self.validate_epoch(epoch)



    def train_epoch(self, epoch):
        pass


    def validate_epoch(self, epoch):
        pass
