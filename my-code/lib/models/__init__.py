# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:50
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.optim as optim

import lib.utils as utils

from .DenseVNet import DenseVNet
from .UNet3D import UNet3D
from .VNet import VNet
from .AttentionUNet import AttentionU_Net
from .R2UNet import R2U_Net
from .R2AttentionUNet import R2AttentionU_Net
from .HighResNet3D import HighResNet3D
from .DenseVoxelNet import DenseVoxelNet
from .MultiResUNet3D import MultiResUNet3D

from .PMFSNet import PMFSNet



def get_model_optimizer_lr_scheduler(opt):
    # 初始化网络模型
    if opt["model_name"] == "DenseVNet":
        model = DenseVNet(in_channels=opt["in_channels"], classes=opt["classes"])

    elif opt["model_name"] == "UNet3D":
        model = UNet3D(opt["in_channels"], opt["classes"], final_sigmoid=False)

    elif opt["model_name"] == "VNet":
        model = VNet(in_channels=opt["in_channels"], classes=opt["classes"])

    elif opt["model_name"] == "AttentionUNet":
        model = AttentionU_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

    elif opt["model_name"] == "R2UNet":
        model = R2U_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

    elif opt["model_name"] == "R2AttentionUNet":
        model = R2AttentionU_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

    elif opt["model_name"] == "HighResNet3D":
        model = HighResNet3D(in_channels=opt["in_channels"], classes=opt["classes"])

    elif opt["model_name"] == "DenseVoxelNet":
        model = DenseVoxelNet(in_channels=opt["in_channels"], classes=opt["classes"])

    elif opt["model_name"] == "MultiResUNet3D":
        model = MultiResUNet3D(in_channels=opt["in_channels"], classes=opt["classes"])

    elif opt["model_name"] == "PMRFNet":
        model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"])

    else:
        raise RuntimeError(f"{opt['model_name']}是不支持的网络模型！")


    # 把模型放到GPU上
    model = model.to(opt["device"])

    # 随机初始化模型参数
    utils.init_weights(model, init_type="kaiming")


    # 初始化优化器
    if opt["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"],
                              weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"],
                                  momentum=opt["momentum"])

    elif opt["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    else:
        raise RuntimeError(
            f"{opt['optimizer_name']}是不支持的优化器！")

    # 初始化学习率调度器
    if opt["lr_scheduler_name"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt["milestones"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt["T_max"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt["T_0"],
                                                                      T_mult=opt["T_mult"])

    elif opt["lr_scheduler_name"] == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt["learning_rate"],
                                                     steps_per_epoch=opt["steps_per_epoch"], epochs=opt["end_epoch"], cycle_momentum=False)

    elif opt["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt["mode"], factor=opt["factor"],
                                                            patience=opt["patience"])
    else:
        raise RuntimeError(
            f"{opt['lr_scheduler_name']}是不支持的学习率调度器！")

    return model, optimizer, lr_scheduler

