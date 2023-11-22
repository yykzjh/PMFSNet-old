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

from lib import utils, dataloaders, models, losses, metrics, trainers



# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "1",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resize_shape": (224, 224),  # 图像resize大小

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    "augmentation_p": 0.35282704987170166,  # 每张图像做数据增强的概率

    "color_jitter": 0.2680656935993321,  # 亮度、对比度、饱和度变化率范围

    "random_rotation_angle": 15,  # 随机旋转角度范围

    # 标准化均值
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "ISIC2018",  # 数据集名称， 可选["NCTooth", "MMOTU", "ISIC2018"]

    "dataset_path": r"./datasets/ISIC2018_only_use_trainset",  # 数据集路径

    "batch_size": 32,  # batch_size大小

    "num_workers": 2,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMFSNet",  # 模型名称["PMFSNet", "MobileNetV2", "UNet", "MsRED", "CKDNet", "BCDUNet", "CANet", "CENet", "CPFNet"]

    "in_channels": 3,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
    {
        0: "background",
        1: "foreground"
    },

    "resume": None,  # 是否重启之前某个训练节点，继续训练;如果需要则指定.state文件路径

    "pretrain": r"../../PMFS-Net/my-2d/pretrain/PMFSNet2D-basic_global.pth",  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "AdamW",  # 优化器名称，可选["SGD", "Adagrad", "RMSprop", "Adam", "AdamW", "Adamax", "Adadelta"]

    "learning_rate": 0.005,  # 学习率

    "weight_decay": 0.000001,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.9657205586290213,  # 动量大小

    # ———————————————————————————————————————————    学习率调度器     —————————————————————————————————————————————————————

    "lr_scheduler_name": "CosineAnnealingWarmRestarts",  # 学习率调度器名称，可选["ExponentialLR", "StepLR", "MultiStepLR",
    # "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "OneCycleLR", "ReduceLROnPlateau"]

    "gamma": 0.9582311026945434,  # 学习率衰减系数

    "step_size": 20,  # StepLR的学习率衰减步长

    "milestones": [1, 3, 5, 7, 8, 9],  # MultiStepLR的学习率衰减节点列表

    "T_max": 100,  # CosineAnnealingLR的半周期

    "T_0": 20,  # CosineAnnealingWarmRestarts的周期

    "T_mult": 3,  # CosineAnnealingWarmRestarts的周期放大倍数

    "mode": "max",  # ReduceLROnPlateau的衡量指标变化方向

    "patience": 20,  # ReduceLROnPlateau的衡量指标可以停止优化的最长epoch

    "factor": 0.3,  # ReduceLROnPlateau的衰减系数

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "metric_names": ["DSC", "IoU", "JI", "ACC"],  # 评价指标，可选["DSC", "IoU", "JI", "ACC"]

    "loss_function_name": "DiceLoss",  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss",
    # "MSELoss","SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

    "class_weight": [0.07033986265670567, 1-0.07033986265670567],  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "dice_loss_mode": "extension",  # Dice Loss的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    "dice_mode": "standard",  # DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "optimize_params": True,  # 程序是否处于优化参数的模型，不需要保存训练的权重和中间结果

    "run_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 0,  # 训练时的起始epoch
    "end_epoch": 20,  # 训练时的结束epoch

    "best_metric": 0,  # 保存检查点的初始条件

    "terminal_show_freq": 15,  # 终端打印统计信息的频率,以step为单位

    "save_epoch_freq": 30,  # 每多少个epoch保存一次训练状态和模型参数
}



if __name__ == '__main__':

    if params["optimize_params"]:
        # 获得下一组搜索空间中的参数
        tuner_params = nni.get_next_parameter()
        tuner_params["class_weight"] = [tuner_params["class_weight"], 1 - tuner_params["class_weight"]]
        # 更新参数
        params.update(tuner_params)

    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # 获取GPU设备
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("完成初始化配置")

    # 初始化数据加载器
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("完成初始化数据加载器")

    # 初始化模型、优化器和学习率调整器
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("完成初始化模型:{}、优化器:{}和学习率调整器:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    # 初始化损失函数
    loss_function = losses.get_loss_function(params)
    print("完成初始化损失函数")

    # 初始化各评价指标
    metric = metrics.get_metric(params)
    print("完成初始化评价指标")

    # 初始化训练器
    trainer = trainers.Trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)

    # 如果需要继续训练或者加载预训练权重
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()

    # 开始训练
    trainer.training()

















