# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/19 16:28
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob
import numpy as np

from torch.utils.data import Dataset

import lib.transforms as transforms
import lib.utils as utils


class NCToothDataset(Dataset):
    """
    读取nrrd牙齿数据集
    """

    def __init__(self, opt, mode):
        """
        Args:
            opt: 参数字典
            mode: train/val
        """
        self.mode = mode
        self.run_dir = opt["run_dir"]
        self.root = opt["dataset_path"]
        self.train_path = os.path.join(self.root, "train")
        self.val_path = os.path.join(self.root, "val")
        self.augmentations = [
            opt["open_elastic_transform"], opt["open_gaussian_noise"], opt["open_random_flip"],
            opt["open_random_rescale"], opt["open_random_rotate"], opt["open_random_shift"]]
        self.sub_volume_root_dir = os.path.join(self.root, "sub_volumes")
        self.data = []
        self.transform = None

        # 分类创建子卷数据集
        if self.mode == 'train':
            # 定义数据增强
            all_augments = [
                 transforms.ElasticTransform(alpha=opt["elastic_transform_alpha"],
                                             sigma=opt["elastic_transform_sigma"]),
                 transforms.GaussianNoise(mean=opt["gaussian_noise_mean"],
                                          std=opt["gaussian_noise_std"]),
                 transforms.RandomFlip(),
                 transforms.RandomRescale(min_percentage=opt["random_rescale_min_percentage"],
                                          max_percentage=opt["random_rescale_max_percentage"]),
                 transforms.RandomRotation(min_angle=opt["random_rotate_min_angle"],
                                           max_angle=opt["random_rotate_max_angle"]),
                 transforms.RandomShift(max_percentage=opt["random_shift_max_percentage"])
            ]
            # 获取实际要进行的数据增强
            practice_augments = [all_augments[i] for i, is_open in enumerate(self.augmentations) if is_open]
            # 定义数据增强方式
            if opt["augmentation_method"] == "Choice":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.RandomAugmentChoice(practice_augments, p=opt["augmentation_probability"]),
                    transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    # transforms.ToTensor(),
                    transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
                ])
            elif opt["augmentation_method"] == "Compose":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ComposeAugments(practice_augments, p=opt["augmentation_probability"]),
                    transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    # transforms.ToTensor(),
                    transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
                ])

            # 定义子卷训练集目录
            self.sub_volume_train_dir = os.path.join(self.sub_volume_root_dir, "train")
            # 定义子卷原图像存储目录
            self.sub_volume_images_dir = os.path.join(self.sub_volume_train_dir, "images")
            # 定义子卷标注图像存储目录
            self.sub_volume_labels_dir = os.path.join(self.sub_volume_train_dir, "labels")

            # 创建子卷训练集目录
            utils.make_dirs(self.sub_volume_train_dir)
            # 创建子卷原图像存储目录
            utils.make_dirs(self.sub_volume_images_dir)
            # 创建子卷标注图像存储目录
            utils.make_dirs(self.sub_volume_labels_dir)

            # 获取数据集中所有原图图像和标注图像的路径
            images_path_list = sorted(glob.glob(os.path.join(self.train_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.train_path, "labels", "*.nrrd")))

            # 生成子卷数据集
            self.data = utils.create_sub_volumes(images_path_list, labels_path_list, opt["samples_train"],
                                                 opt["resample_spacing"], opt["clip_lower_bound"],
                                                 opt["clip_upper_bound"], opt["crop_size"],
                                                 opt["crop_threshold"], self.sub_volume_train_dir)

        elif self.mode == 'val':
            # 定义验证集数据增强
            self.val_transforms = transforms.ComposeTransforms([
                transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                # transforms.ToTensor(),
                transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
            ])

            # 定义子卷验证集目录
            self.sub_volume_val_dir = os.path.join(self.sub_volume_root_dir, "val")
            # 定义子卷原图像存储目录
            self.sub_volume_images_dir = os.path.join(self.sub_volume_val_dir, "images")
            # 定义子卷标注图像存储目录
            self.sub_volume_labels_dir = os.path.join(self.sub_volume_val_dir, "labels")

            # 创建子卷训练集目录
            utils.make_dirs(self.sub_volume_val_dir)
            # 创建子卷原图像存储目录
            utils.make_dirs(self.sub_volume_images_dir)
            # 创建子卷标注图像存储目录
            utils.make_dirs(self.sub_volume_labels_dir)

            # 获取数据集中所有原图图像和标注图像的路径
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nrrd")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nrrd")))

            # 对验证集进行预处理
            self.data = utils.preprocess_val_dataset(images_path_list, labels_path_list, opt["resample_spacing"],
                                                     opt["clip_lower_bound"], opt["clip_upper_bound"],
                                                     self.sub_volume_val_dir)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image_path, label_path = self.data[index]
        image, label = np.load(image_path), np.load(label_path)

        if self.mode == 'train':
            transform_image, transform_label = self.train_transforms(image, label)
            return transform_image.unsqueeze(0), transform_label

        else:
            transform_image, transform_label = self.val_transforms(image, label)
            return transform_image.unsqueeze(0), transform_label
