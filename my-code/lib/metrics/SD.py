# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/6/13 2:54
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import math
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append(r"D:\Projects\Python\3D-tooth-segmentation\PMFS-Net：Polarized Multi-scale Feature Self-attention Network For CBCT Tooth Segmentation\my-code")
from lib.utils import *



class SurfaceDice(object):
    def __init__(self, num_classes=33, theta=1.0, sigmoid_normalization=False):
        """
        定义表面Dice系数(SD)评价指标计算器

        :param num_classes: 类别数
        :param theta: 判断两个点处于相同位置的最大距离
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(SurfaceDice, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        self.theta = theta
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target):
        """
        表面Dice系数(SO)

        :param input: 网络模型输出的预测图,(B, C, H, W, D)
        :param target: 标注图像,(B, H, W, D)
        :return:
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"
        # 转换seg和target数据类型为整型
        seg = seg.type(torch.uint8)
        target = target.type(torch.uint8)

        return compute_per_channel_so(seg, target, self.num_classes, theta=self.theta)




if __name__ == '__main__':
    pred = torch.randn((4, 33, 32, 32, 16))
    gt = torch.randint(33, (4, 32, 32, 16))

    SD_metric = SurfaceDice(num_classes=33, theta=1.0)

    SD_per_channel = SD_metric(pred, gt)

    print(SD_per_channel)




















