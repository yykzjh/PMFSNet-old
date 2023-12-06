import torch
import torch.optim as optim

import lib.utils as utils

from .PMFSNet import PMFSNet
from .MobileNetV2 import MobileNetV2
from .PSPNet import PSPNet
from .DANet import DANet
from .SegFormer import SegFormer
from .UNet import UNet
from .TransUNet import TransUNet
from .TransUNet import CONFIGS as CONFIGS_ViT_seg
from .BiSeNetV2 import BiSeNetV2


def get_model_optimizer_lr_scheduler(opt):
    # 初始化网络模型
    if opt["model_name"] == "PMFSNet":
        model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"])

    elif opt["model_name"] == "MobileNetV2":
        model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

    elif opt["model_name"] == "PSPNet":
        model = PSPNet(n_classes=opt["classes"], backend='resnet50', pretrained=True)

    elif opt["model_name"] == "DANet":
        model = DANet(nclass=opt["classes"])

    elif opt["model_name"] == "SegFormer":
        model = SegFormer(channels=opt["in_channels"], num_classes=opt["classes"])

    elif opt["model_name"] == "UNet":
        model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

    elif opt["model_name"] == "TransUNet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = opt["classes"]
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(opt["resize_shape"][0] / 16), int(opt["resize_shape"][1] / 16))
        model = TransUNet(config_vit, img_size=opt["resize_shape"][0], num_classes=config_vit.n_classes)

    elif opt["model_name"] == "BiSeNetV2":
        model = BiSeNetV2(n_classes=opt["classes"])

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

    elif opt["model_name"] == "PSPNet":
        model = PSPNet(n_classes=opt["classes"], backend='resnet50', pretrained=False)

    elif opt["model_name"] == "DANet":
        model = DANet(nclass=opt["classes"])

    elif opt["model_name"] == "SegFormer":
        model = SegFormer(channels=opt["in_channels"], num_classes=opt["classes"])

    elif opt["model_name"] == "UNet":
        model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

    elif opt["model_name"] == "TransUNet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = opt["classes"]
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(opt["resize_shape"][0] / 16), int(opt["resize_shape"][1] / 16))
        model = TransUNet(config_vit, img_size=opt["resize_shape"][0], num_classes=config_vit.n_classes)

    elif opt["model_name"] == "BiSeNetV2":
        model = BiSeNetV2(n_classes=opt["classes"])

    else:
        raise RuntimeError(f"{opt['model_name']}是不支持的网络模型！")

    # 把模型放到GPU上
    model = model.to(opt["device"])

    return model