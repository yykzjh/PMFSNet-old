# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/29 01:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob

import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms as my_transforms


class ISIC2018Dataset(Dataset):
    """
    读取ISIC2018数据集
    """

    def __init__(self, opt, mode):
        """
        初始化ISIC2018数据集
        :param opt: 超参数字典
        :param mode: train/valid
        """
        super(ISIC2018Dataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_dir = os.path.join(self.root, "train")
        self.valid_dir = os.path.join(self.root, "test")
        # 定义数据增强方法
        self.transforms_dict = {
            "train": my_transforms.Compose([
                my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ]),
            "valid": my_transforms.Compose([
                my_transforms.Resize(self.opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ])
        }

        # 获取并存储数据集所有图像路径
        if mode == "train":
            self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.JPG")))
            self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "labels", "*.PNG")))
        else:
            self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.JPG")))
            self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "labels", "*.PNG")))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        # 读取图像
        image = cv2.imread(self.images_list[index], -1)
        label = cv2.imread(self.labels_list[index], -1)
        # 数据预处理和数据增强
        image, label = self.transforms_dict[self.mode](image, label)

        return image, label


