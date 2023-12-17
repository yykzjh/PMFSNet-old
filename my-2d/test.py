# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:52
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import cv2
import glob
import math
from tqdm import tqdm
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
import lib.transforms as my_transforms



# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resize_shape": (224, 224),  # 图像resize大小

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    # 标准化均值
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMFSNet",  # 模型名称["PMFSNet", "MobileNetV2", "UNet", "MsRED", "CKDNet", "BCDUNet", "CANet", "CENet", "CPFNet", "AttU_Net"]

    "in_channels": 3,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
    {
        0: "background",
        1: "foreground"
    },

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径
}


def segment_image(model, image_path, label_path):
    transform = my_transforms.Compose([
        my_transforms.Resize(params["resize_shape"]),
        my_transforms.ToTensor(),
        my_transforms.Normalize(mean=params["normalize_means"], std=params["normalize_stds"])
    ])
    # 读取图像
    image = cv2.imread(image_path, -1)
    label = cv2.imread(label_path, -1)
    label[label == 255] = 1
    # 数据预处理和数据增强
    image, label = transform(image, label)
    # image扩充一维
    image = torch.unsqueeze(image, dim=0)
    # 转换数据格式
    label = label.to(dtype=torch.uint8)
    # 放到cuda上

    # 预测分割
    pred = torch.squeeze(model(image.to(params["device"])), dim=0)
    segmented_image_np = torch.argmax(pred, dim=0).to(dtype=torch.uint8).cpu().numpy()
    label_np = label.numpy()
    # image和numpy扩展到三维
    seg_image = np.dstack([segmented_image_np] * 3)
    label = np.dstack([label_np] * 3)
    # 定义红色、白色和绿色图像
    red = np.zeros((224, 224, 3))
    red[:, :, 0] = 255
    green = np.zeros((224, 224, 3))
    green[:, :, 1] = 255
    white = np.ones((224, 224, 3)) * 255
    segmented_display_image = np.zeros((224, 224, 3))
    segmented_display_image = np.where(seg_image & label, white, segmented_display_image)
    segmented_display_image = np.where(seg_image & ~label, red, segmented_display_image)
    segmented_display_image = np.where(~seg_image & label, green, segmented_display_image)
    return segmented_display_image


def generate_segment_result_images(model_names):
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

    # 遍历模型
    for j, model_name in tqdm(enumerate(model_names)):
        params["model_name"] = model_name
        params["pretrain"] = os.path.join(r"./pretrain", model_name + ".pth")
        # 初始化模型
        model = models.get_model(params)
        print("完成初始化模型:{}".format(params["model_name"]))
        # 加载模型权重
        pretrain_state_dict = torch.load(params["pretrain"], map_location=lambda storage, loc: storage.cuda(params["device"]))
        model_state_dict = model.state_dict()
        load_count = 0  # 成功加载参数计数
        for param_name in model_state_dict.keys():
            if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                load_count += 1
        model.load_state_dict(model_state_dict, strict=True)
        print("{:.2f}%的模型参数成功加载预训练权重".format(100 * load_count / len(model_state_dict)))
        # 遍历所有图
        for i in tqdm(range(4)):
            image_path = os.path.join(r"./images/ISIC2018_segment_result", str(i) + "0.jpg")
            label_path = os.path.join(r"./images/ISIC2018_segment_result", str(i) + "1.png")
            segmented_display_image = segment_image(model, image_path, label_path)
            segmented_display_image = segmented_display_image[:, :, ::-1]
            cv2.imwrite(os.path.join(r"./images/ISIC2018_segment_result", str(i) + str(j+2) + ".jpg"), segmented_display_image)




if __name__ == '__main__':
    generate_segment_result_images(["UNet", "AttU_Net", "CANet", "BCDUNet", "CENet", "CPFNet", "CKDNet", "PMFSNet"])
























