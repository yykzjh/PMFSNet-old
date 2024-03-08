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
import random
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

from lib import utils
from lib import dataloaders
from lib import models
from lib import testers
from lib import metrics

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

    "clip_lower_bound": -1412,  # clip的下边界数值
    "clip_upper_bound": 17943,  # clip的上边界数值

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    # 标准化均值
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMFSNet",  # 模型名称，可选["DenseVNet","UNet3D", "VNet", "AttentionUNet3D", "R2UNet", "R2AttentionUNet",
    # "HighResNet3D", "DenseVoxelNet", "MultiResUNet3D", "DenseASPPUNet", "PMFSNet", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
    {
        0: "background",
        1: "foreground"
    },

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    # ———————————————————————————————————————————    学习率调度器     —————————————————————————————————————————————————————

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "metric_names": ["DSC"],  # 采用除了dsc之外的评价指标，可选["DSC","ASSD","HD"]

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [32, 32, 32]
}


def get_analyse_image_and_dsc(seg_image, label):
    label[label == 255] = 1
    dsc_score = utils.cal_dsc(seg_image, label)
    seg_image = np.dstack([seg_image] * 3)
    label = np.dstack([label] * 3)
    # 定义红色、白色和绿色图像
    red = np.zeros_like(seg_image)
    red[:, :, 0] = 255
    green = np.zeros_like(seg_image)
    green[:, :, 1] = 255
    white = np.ones_like(seg_image) * 255
    segmented_display_image = np.zeros_like(seg_image)
    segmented_display_image = np.where(seg_image & label, white, segmented_display_image)
    segmented_display_image = np.where(seg_image & ~label, red, segmented_display_image)
    segmented_display_image = np.where(~seg_image & label, green, segmented_display_image)

    return segmented_display_image, dsc_score


def generate_segment_result_images(model_names):
    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])
    # 获取GPU设备
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("完成初始化配置")

    # 初始化一些路径
    dataset_root_dir = r"/data01/zjh/cbct-tooth-segmentation/binary-code/datasets/NC-release-data-checked/valid"
    images_dir = os.path.join(dataset_root_dir, "images")
    labels_dir = os.path.join(dataset_root_dir, "labels")
    cnt = 0
    # 遍历所有图像
    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name)
        image = utils.load_image_or_label(image_path, params["resample_spacing"], type="image")
        label = utils.load_image_or_label(label_path, params["resample_spacing"], type="label")
        label[label == 1] = 255
        # 暂存所有模型的分割结果
        segmented_images_list = []
        # 遍历所有模型
        for model_name in model_names:
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
            # 分割
            tester = testers.Tester(params, model, None)
            segmented_image = tester.test_single_image_without_label(image.copy())
            segmented_images_list.append(segmented_image)
        # 遍历每张图像所有slice
        for slice_ind in range(0, label.shape[2]):
            max_dsc_score = 0
            max_model_name = None
            segment_result_slices_list = []
            for j, segmented_image in enumerate(segmented_images_list):
                segment_result_slice, dsc_score = get_analyse_image_and_dsc(segmented_image[:, :, slice_ind].copy(), label[:, :, slice_ind].copy())
                segment_result_slices_list.append(segment_result_slice)
                if dsc_score > max_dsc_score:
                    max_dsc_score = dsc_score
                    max_model_name = model_names[j]
            if max_model_name == "PMFSNet":
                # image_slice = cv2.resize(image[:, :, slice_ind], (224, 224), interpolation=cv2.INTER_AREA)
                # image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                # image_slice *= 255
                # image_slice = image_slice.astype(np.uint8)
                # cv2.imwrite(os.path.join(r"./images/NC-release-data_segment_result", "{:04d}".format(cnt) + "_00.jpg"), image_slice)
                label_slice = cv2.resize(label[:, :, slice_ind], (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(r"./images/NC-release-data_segment_result", "{:04d}".format(cnt) + "_00.jpg"), label_slice)
                for j, segment_result_slice in enumerate(segment_result_slices_list):
                    segment_result_slice = cv2.resize(segment_result_slice, (224, 224), interpolation=cv2.INTER_NEAREST)
                    segment_result_slice = segment_result_slice[:, :, ::-1]
                    cv2.imwrite(os.path.join(r"./images/NC-release-data_segment_result", "{:04d}".format(cnt) + "_" + "{:02d}".format(j + 1) + ".jpg"), segment_result_slice)
                cnt += 1

    # 初始化一些路径
    dataset_root_dir = r"/data01/zjh/cbct-tooth-segmentation/binary-code/datasets/NC-release-data-checked/train"
    images_dir = os.path.join(dataset_root_dir, "images")
    labels_dir = os.path.join(dataset_root_dir, "labels")
    cnt = 0
    # 遍历所有图像
    for image_name in tqdm(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, image_name)
        image = utils.load_image_or_label(image_path, params["resample_spacing"], type="image")
        label = utils.load_image_or_label(label_path, params["resample_spacing"], type="label")
        label[label == 1] = 255
        # 暂存所有模型的分割结果
        segmented_images_list = []
        # 遍历所有模型
        for model_name in model_names:
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
            # 分割
            tester = testers.Tester(params, model, None)
            segmented_image = tester.test_single_image_without_label(image.copy())
            segmented_images_list.append(segmented_image)
        # 遍历每张图像所有slice
        for slice_ind in range(0, label.shape[2]):
            max_dsc_score = 0
            max_model_name = None
            segment_result_slices_list = []
            for j, segmented_image in enumerate(segmented_images_list):
                segment_result_slice, dsc_score = get_analyse_image_and_dsc(segmented_image[:, :, slice_ind].copy(), label[:, :, slice_ind].copy())
                segment_result_slices_list.append(segment_result_slice)
                if dsc_score > max_dsc_score:
                    max_dsc_score = dsc_score
                    max_model_name = model_names[j]
            if max_model_name == "PMFSNet":
                # image_slice = cv2.resize(image[:, :, slice_ind], (224, 224), interpolation=cv2.INTER_AREA)
                # image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                # image_slice *= 255
                # image_slice = image_slice.astype(np.uint8)
                # cv2.imwrite(os.path.join(r"./images/NC-release-data_segment_result", "{:04d}".format(cnt) + "_00.jpg"), image_slice)
                label_slice = cv2.resize(label[:, :, slice_ind], (224, 224), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(r"./images/NC-release-data_segment_result", "{:04d}".format(cnt) + "_00.jpg"), label_slice)
                for j, segment_result_slice in enumerate(segment_result_slices_list):
                    segment_result_slice = cv2.resize(segment_result_slice, (224, 224), interpolation=cv2.INTER_NEAREST)
                    segment_result_slice = segment_result_slice[:, :, ::-1]
                    cv2.imwrite(os.path.join(r"./images/NC-release-data_segment_result", "{:04d}".format(cnt) + "_" + "{:02d}".format(j + 1) + ".jpg"), segment_result_slice)
                cnt += 1


if __name__ == '__main__':
    generate_segment_result_images(["UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet", "PMFSNet"])
