"""
Implementation of BaseModel taken and modified from here
https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/basemodel/basemodel.py
"""

import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "Model"


    @abstractmethod
    def forward(self, x):
        pass


    @abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass


    @property
    def device(self):
        return next(self.parameters()).device


    def load_checkpoint(self, ckpt_file_path):
        """
        加载检查点
        Args:
            ckpt_file_path: 检查点文件路径

        Returns:

        """
        if not os.path.exists(ckpt_file_path):
            raise RuntimeError("no checkpoint file in {}".format(ckpt_file_path))

        try:
            ckpt_dict = torch.load(ckpt_file_path)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file_path, map_location=lambda storage, loc: storage)

        return ckpt_dict


    def save_checkpoint(self, save_path, epoch, type="last", best_metric=None, optimizer=None, metric_name=None):
        """
        保存检查点
        Args:
            save_path: 存储路径
            epoch: 迭代次数
            type: 检查点类型best/last
            best_metric: 当前衡量指标最优值
            optimizer: 优化器
            metric_name: 衡量指标名

        Returns:

        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 创建检查点字典
        ckpt_dict = {
            'epoch': epoch,
            'best_metric': best_metric,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
        }

        # 根据检查点类型生成文件名
        if type == "last":
            ckpt_file_name = "{}_{}_epoch.pth".format(self.model_name, type)
        else:
            ckpt_file_name = "{}_{}_{}_{}.pth".format(self.model_name, type, metric_name, best_metric)

        torch.save(ckpt_dict, os.path.join(save_path, ckpt_file_name))


    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)

        return num_total_params, num_trainable_params


    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()
