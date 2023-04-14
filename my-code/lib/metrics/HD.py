import numpy as np



class HausdorffDistance(object):
    def __init__(self, c=6, num_classes=33):
        """
        定义豪斯多夫距离评价指标计算器
        Args:
            c: 连通度
            num_classes: 类别数
        """
        super(HausdorffDistance, self).__init__()
        # 初始化参数
        self.c = c
        self.num_classes = num_classes


    def calculate(self, pred, gt):
        """
        计算豪斯多夫距离评价指标
        Args:
            pred: 经过softmax后的预测图像
            gt: ground truth 图像

        Returns: 各batch中各类别牙齿的平均豪斯多夫距离
        """
        











