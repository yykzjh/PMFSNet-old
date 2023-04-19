# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/8 17:09
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import numpy as np



class ToTensor(object):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound



    def __call__(self, img_numpy, label):
        """
        Args:
            img_numpy: Image transforms from numpy to tensor
            label: Label segmentation map transforms from numpy to tensor

        Returns:
        """
        # 转换为tensor
        img_tensor = torch.FloatTensor(np.ascontiguousarray(img_numpy))
        label_tensor = torch.FloatTensor(np.ascontiguousarray(label))
        # 将图像灰度值归一化到0~1
        img_tensor = img_tensor / (self.upper_bound - self.lower_bound)
        return img_tensor, label_tensor


# class ToTensor(object):
#     def __init__(self):
#         pass
#
#
#
#     def __call__(self, img_numpy, label):
#         """
#         Args:
#             img_numpy: Image transforms from numpy to tensor
#             label: Label segmentation map transforms from numpy to tensor
#
#         Returns:
#         """
#         # 转换为tensor
#         img_tensor = torch.FloatTensor(np.ascontiguousarray(img_numpy))
#         label_tensor = torch.FloatTensor(np.ascontiguousarray(label))
#         # 获取最大值和最小值
#         min_val, max_val = img_tensor.min(), img_tensor.max()
#         # 将图像灰度值归一化到0~1
#         img_tensor = (img_tensor - min_val) / (max_val - min_val)
#         return img_tensor, label_tensor




