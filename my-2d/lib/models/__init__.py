import torch
import torch.optim as optim

import lib.utils as utils

from .PMFSNet import PMFSNet
from .MobileNetV2 import MobileNetV2
from .UNet import UNet
from .MsRED import Ms_red_v1, Ms_red_v2
from .CKDNet import DeepLab_Aux
from .BCDUNet import BCDUNet
from .CANet import Comprehensive_Atten_Unet
from .CENet import CE_Net
from .CPFNet import CPF_Net
from .AttU_Net import AttU_Net


def get_model_optimizer_lr_scheduler(opt):
    # 初始化网络模型
    if opt["model_name"] == "PMFSNet":
        model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"])

    elif opt["model_name"] == "MobileNetV2":
        model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

    elif opt["model_name"] == "UNet":
        model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

    elif opt["model_name"] == "MsRED":
        model = Ms_red_v2(classes=opt["classes"], channels=opt["in_channels"], out_size=opt["resize_shape"])

    elif opt["model_name"] == "CKDNet":
        model = DeepLab_Aux(num_classes=opt["classes"])

    elif opt["model_name"] == "BCDUNet":
        model = BCDUNet(output_dim=opt["classes"], input_dim=opt["in_channels"], frame_size=opt["resize_shape"])

    elif opt["model_name"] == "CANet":
        model = Comprehensive_Atten_Unet(in_ch=opt["in_channels"], n_classes=opt["classes"], out_size=opt["resize_shape"])

    elif opt["model_name"] == "CENet":
        model = CE_Net(classes=opt["classes"], channels=opt["in_channels"])

    elif opt["model_name"] == "CPFNet":
        model = CPF_Net(classes=opt["classes"], channels=opt["in_channels"])

    elif opt["model_name"] == "AttU_Net":
        model = AttU_Net(img_ch=opt["in_channels"], output_ch=opt["classes"])

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

    elif opt["optimizer_name"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

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



def get_model(opt):
    # 初始化网络模型
    if opt["model_name"] == "PMFSNet":
        model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"])

    elif opt["model_name"] == "MobileNetV2":
        model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

    elif opt["model_name"] == "UNet":
        model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

    elif opt["model_name"] == "MsRED":
        model = Ms_red_v2(classes=opt["classes"], channels=opt["in_channels"], out_size=opt["resize_shape"])

    elif opt["model_name"] == "CKDNet":
        model = DeepLab_Aux(num_classes=opt["classes"])

    elif opt["model_name"] == "BCDUNet":
        model = BCDUNet(output_dim=opt["classes"], input_dim=opt["in_channels"], frame_size=opt["resize_shape"])

    elif opt["model_name"] == "CANet":
        model = Comprehensive_Atten_Unet(in_ch=opt["in_channels"], n_classes=opt["classes"], out_size=opt["resize_shape"])

    elif opt["model_name"] == "CENet":
        model = CE_Net(classes=opt["classes"], channels=opt["in_channels"])

    elif opt["model_name"] == "CPFNet":
        model = CPF_Net(classes=opt["classes"], channels=opt["in_channels"])

    elif opt["model_name"] == "AttU_Net":
        model = AttU_Net(img_ch=opt["in_channels"], output_ch=opt["classes"])

    else:
        raise RuntimeError(f"{opt['model_name']}是不支持的网络模型！")

    # 把模型放到GPU上
    model = model.to(opt["device"])

    return model