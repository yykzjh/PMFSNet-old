import math
import torch
import torch.nn as nn
import numpy as np

from lib.utils import *



class AverageSymmetricSurfaceDistance(object):
    def __init__(self, num_classes=33, c=6, sigmoid_normalization=False):
        """
        定义平均对称表面距离(ASSD)评价指标计算器
        Args:
            num_classes: 类别数
            c: 连通度
            sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(AverageSymmetricSurfaceDistance, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        self.c = c
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target):
        """
        平均对称表面距离(ASSD)
        Args:
            input: 网络模型输出的预测图,(B, C, H, W, D)
            target: 标注图像,(B, H, W, D)

        Returns:
        """
        # ont-hot处理，将标注图在axis=1维度上扩张，该维度大小等于预测图的通道C大小，维度上每一个索引依次对应一个类别,(B, C, H, W, D)
        target = expand_as_one_hot(target.long(), self.num_classes)

        # 判断one-hot处理后标注图和预测图的维度是否都是5维
        assert input.dim() == target.dim() == 5, "one-hot处理后标注图和预测图的维度不是都为5维！"
        # 判断one-hot处理后标注图和预测图的尺寸是否一致
        assert input.size() == target.size(), "one-hot处理后预测图和标注图的尺寸不一致！"

        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        return compute_per_channel_assd(input, target, c=self.c)














if __name__ == '__main__':
    pred = torch.randn((4, 33, 32, 32, 16))
    gt = torch.randint(33, (4, 32, 32, 16))

    ASSD_metric = AverageSymmetricSurfaceDistance(c=6, num_classes=33)

    batch_ASSD = ASSD_metric(pred, gt)

    print(batch_ASSD)




















