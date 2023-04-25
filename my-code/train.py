# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:52
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import glob
import math
import tqdm
import shutil
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import nni

from lib import utils, dataloaders, models, losses, trainers



# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": -3566,  # clip的下边界数值
    "clip_upper_bound": 14913,  # clip的上边界数值

    "samples_train": 4,  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    "crop_threshold": 0.0,  # 随机裁剪时需要满足的条件，不满足则重新随机裁剪的位置。条件表示的是裁剪区域中的前景占原图总前景的最小比例

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    "augmentation_probability": 0.3,  # 每张图像做数据增强的概率
    "augmentation_method": "Choice",  # 数据增强的方式，可选["Compose", "Choice"]

    # 弹性形变参数
    "open_elastic_transform": True,  # 是否开启弹性形变数据增强
    "elastic_transform_sigma": 20,  # 高斯滤波的σ,值越大，弹性形变越平滑
    "elastic_transform_alpha": 1,  # 形变的幅度，值越大，弹性形变越剧烈

    # 高斯噪声参数
    "open_gaussian_noise": True,  # 是否开启添加高斯噪声数据增强
    "gaussian_noise_mean": 0,  # 高斯噪声分布的均值
    "gaussian_noise_std": 0.01,  # 高斯噪声分布的标准差,值越大噪声越强

    # 随机翻转参数
    "open_random_flip": True,  # 是否开启随机翻转数据增强

    # 随机缩放参数
    "open_random_rescale": True,  # 是否开启随机缩放数据增强
    "random_rescale_min_percentage": 0.5,  # 随机缩放时,最小的缩小比例
    "random_rescale_max_percentage": 1.5,  # 随机缩放时,最大的放大比例

    # 随机旋转参数
    "open_random_rotate": True,  # 是否开启随机旋转数据增强
    "random_rotate_min_angle": -50,  # 随机旋转时,反方向旋转的最大角度
    "random_rotate_max_angle": 50,  # 随机旋转时,正方向旋转的最大角度

    # 随机位移参数
    "open_random_shift": True,  # 是否开启随机位移数据增强
    "random_shift_max_percentage": 0.3,  # 在图像的三个维度(D,H,W)都进行随机位移，位移量的范围为(-0.3×(D、H、W),0.3×(D、H、W))

    # 标准化均值
    "normalize_mean": 0.17375025153160095,
    "normalize_std": 0.053983770310878754,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "NCTooth",  # 数据集名称， 可选["NCTooth", ]

    "dataset_path": r"./datasets/NC-release-data-modify",  # 数据集路径

    "create_data": True,  # 是否重新分割子卷训练集

    "batch_size": 1,  # batch_size大小

    "num_workers": 1,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMRFNet",  # 模型名称，可选["DenseVNet","UNet3D", "VNet", "AttentionUNet", "R2UNet", "R2AttentionUNet", # "PMRFNet"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "resume": None,  # 是否重启之前某个训练节点，继续训练;如果需要则指定checkpoint文件路径

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "Adam",  # 优化器名称，可选["SGD", "Adagrad", "RMSprop", "Adam", "Adamax", "Adadelta"]

    "learning_rate": 0.001,  # 学习率

    "weight_decay": 0.00001,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.9,  # 动量大小

    # ———————————————————————————————————————————    学习率调度器     —————————————————————————————————————————————————————

    "lr_scheduler_name": "ReduceLROnPlateau",  # 学习率调度器名称，可选["ExponentialLR", "StepLR", "MultiStepLR",
    # "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "OneCycleLR", "ReduceLROnPlateau"]

    "gamma": 0.9,  # 学习率衰减系数

    "step_size": 5,  # StepLR的学习率衰减步长

    "milestones": [3, 5, 9, 13, 15, 17, 18, 19],  # MultiStepLR的学习率衰减节点列表

    "T_max": 5,  # CosineAnnealingLR的半周期

    "T_0": 4,  # CosineAnnealingWarmRestarts的周期

    "T_mult": 2,  # CosineAnnealingWarmRestarts的周期放大倍数

    "mode": "min",  # ReduceLROnPlateau的衡量指标变化方向

    "patience": 5,  # ReduceLROnPlateau的衡量指标可以停止优化的最长epoch

    "factor": 0.1,  # ReduceLROnPlateau的衰减系数

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "loss_function_name": "DiceLoss",  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss",
    # "MSELoss","SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

    "class_weight": [0.00002066, 0.00022885, 0.10644896, 0.02451709, 0.03155127, 0.02142642, 0.02350483, 0.02480525,
                     0.01125384, 0.01206108, 0.07426875, 0.02583742, 0.03059388, 0.02485595, 0.02466779, 0.02529981,
                     0.01197175, 0.01272877, 0.16020726, 0.05647514, 0.0285633, 0.01808694, 0.02124704, 0.02175892,
                     0.01802092, 0.01563035, 0., 0.0555509, 0.02747846, 0.01756969, 0.02183707, 0.01934677, 0.01848419,
                     0.01370064, 0.],  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "run_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 1,  # 训练时的起始epoch
    "end_epoch": 20,  # 训练时的结束epoch

    "best_dice": 0.60,  # 保存检查点的初始条件

    "terminal_show_freq": 50,  # 终端打印统计信息的频率,以step为单位

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [4, 4, 4]
}



if __name__ == '__main__':

    # # 获得下一组搜索空间中的参数
    # tuner_params = nni.get_next_parameter()
    # # 更新参数
    # params.update(tuner_params)

    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["cuda"], params["deterministic"], params["benchmark"])

    # 获取GPU设备
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")

    print("完成初始化配置")

    # 初始化数据加载器
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("完成初始化数据加载器")

    # 初始化模型、优化器和学习率调整器
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("完成初始化模型、优化器和学习率调整器")

    # 初始化损失函数
    loss_function = losses.get_loss_function(params)
    print("完成初始化损失函数")

    # 初始化训练器
    # trainer = trainers.Trainer(model, loss_function, optimizer, train_loader, valid_loader, lr_scheduler)

















