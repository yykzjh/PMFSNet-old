# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/7 23:41
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import glob
import tqdm
import shutil
import random
import numpy as np
import trimesh
import json
import nibabel as nib
from tqdm import tqdm
from nibabel.viewers import OrthoSlicer3D
import SimpleITK as sitk
import matplotlib.pyplot as plt

import lib.utils as utils



def load_nii_file(file_path):
    """
    底层读取.nii.gz文件
    Args:
        file_path: 文件路径

    Returns: uint16格式的三维numpy数组, spacing三个元素的元组

    """
    # 读取nii对象
    NiiImage = sitk.ReadImage(file_path)
    # 从nii对象中获取numpy格式的数组，[z, y, x]
    image_numpy = sitk.GetArrayFromImage(NiiImage)
    # 转换维度为 [x, y, z]
    image_numpy = image_numpy.transpose(2, 1, 0)
    # 获取体素间距
    spacing = NiiImage.GetSpacing()

    return image_numpy, spacing


def load_obj_file(file_path):
    """
    底层读取.obj文件
    Args:
        file_path: 文件路径

    Returns:
    """
    obj = trimesh.load(file_path, process=False)
    v = obj.vertices
    f = obj.faces
    v1 = np.array(v)
    f1 = np.array(f)
    print(v1.shape)
    print(f1.shape)

    json_file_path = os.path.splitext(file_path)[0] + ".json"
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    labels = json_file["labels"]
    print(len(labels))


def analyse_dataset(dataset_path):
    # 首先获取所有原始图像和所有标签图像的路径
    images_path_list = glob.glob(os.path.join(dataset_path, "image", "*.nii.gz"))
    labels_path_list = glob.glob(os.path.join(dataset_path, "label", "10*.nii.gz"))

    # 统计数据集一共有多少个类别
    label_set = set()
    for label_path in labels_path_list:
        # 读取标签图像
        label_numpy, spacing = load_nii_file(label_path)
        # 标签值添加到集合中
        label_set = label_set.union(set(np.unique(label_numpy)))
        print(label_path, label_numpy.min(), label_numpy.max())
    print(label_set)


def split_dataset(dataset_dir, train_ratio=0.8, seed=123):
    # 设置随机种子
    np.random.seed(seed)
    # 创建训练集和验证集文件夹
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "valid")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(os.path.join(train_dir, "images"))
    os.makedirs(os.path.join(train_dir, "labels"))
    os.makedirs(os.path.join(val_dir, "images"))
    os.makedirs(os.path.join(val_dir, "labels"))
    # 读取所有的
    images_path_list = glob.glob(os.path.join(dataset_dir, "image", "*.nii.gz"))
    labels_path_list = glob.glob(os.path.join(dataset_dir, "label", "*.nii.gz"))
    # 随机抽样训练集
    trainset_list = random.sample(images_path_list, int(len(images_path_list) * 0.8))
    valset_list = [path for path in images_path_list if path not in trainset_list]
    # 复制训练集
    for path in tqdm(trainset_list):
        file_name = os.path.basename(path)
        dest_image_path = os.path.join(train_dir, "images", file_name)
        # 复制原图
        shutil.copyfile(path, dest_image_path)
        src_label_path = path.replace("image", "label")
        dest_label_path= os.path.join(train_dir, "labels", file_name)
        # 复制标签图
        shutil.copyfile(src_label_path, dest_label_path)
    # 复制验证集
    for path in tqdm(valset_list):
        file_name = os.path.basename(path)
        dest_image_path = os.path.join(val_dir, "images", file_name)
        # 复制原图
        shutil.copyfile(path, dest_image_path)
        src_label_path = path.replace("image", "label")
        dest_label_path = os.path.join(val_dir, "labels", file_name)
        # 复制标签图
        shutil.copyfile(src_label_path, dest_label_path)


def analyse_image_label_consistency(dataset_dir):
    # 定义合格数据集存储根目录
    new_dataset_dir = os.path.join(os.path.dirname(dataset_dir), os.path.basename(dataset_dir) + "-checked")
    new_dataset_image_dir = os.path.join(new_dataset_dir, "image")
    new_dataset_label_dir = os.path.join(new_dataset_dir, "label")
    # 创建新数据集目录
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)
    os.makedirs(new_dataset_dir)
    os.makedirs(new_dataset_image_dir)
    os.makedirs(new_dataset_label_dir)

    # 读取数据集路径列表
    images_list = sorted(glob.glob(os.path.join(dataset_dir, "image", "*.nii.gz")))
    labels_list = sorted(glob.glob(os.path.join(dataset_dir, "label", "*.nii.gz")))
    # 条件一：首先检测image和label数据集大小是否一致
    assert len(images_list) == len(labels_list), "image和label数据集大小不一致"

    # 遍历原images
    for image_path in tqdm(images_list):
        # 首先当前image要有同名的label文件
        virtual_label_path = image_path.replace("image", "label")  # 创建虚拟同名label的路径
        # 条件二：如果没有该label，跳过当前image
        if virtual_label_path not in labels_list:
            continue

        # 如果当前image能找到同名的label，则都先读取
        image_np = utils.load_image_or_label(image_path, [0.5, 0.5, 0.5], type="image")
        label_np = utils.load_image_or_label(virtual_label_path, [0.5, 0.5, 0.5], type="label")
        # 条件三：判断重采样后的image和label图像尺寸是否一致，不一致则跳过当前image
        if image_np.shape != label_np.shape:
            continue

        # 最后分析label标注的前景是否有较大概率是image中的牙齿来判断标注文件的正确性
        # 首先获取image中的最小值和最大值
        min_val = image_np.min()
        max_val = image_np.max()
        # 获取原image的拷贝
        image_copy = image_np.copy()
        # 将image_copy的值归一化到0~255
        image_copy = (image_copy - min_val) / (max_val - min_val) * 255
        # 转换为整型
        image_copy = image_copy.astype(np.int32)
        # 获取骨骼和软组织的阈值
        T0, T1 = utils.get_multi_threshold(image_copy, m=2, max_length=255)
        # 展示分割点
        # plt.hist(image_copy.flatten(), bins=256, color="g", histtype="bar", rwidth=1, alpha=0.6)
        # plt.axvline(T1, color="r")
        # plt.show()
        # 获得阈值在原图中的值
        ori_T = T1 / 255 * (max_val - min_val) + min_val
        # 通过label可知所有前景元素，计算前景部分的值大于ori_T的比例
        ratio = np.sum(image_np[label_np > 0] > ori_T) / np.sum(label_np > 0)
        # 条件四：如果比例小于0.95，则认为当前label图像标注有问题，跳过当前image
        if ratio < 0.90:
            continue

        # 如果前面的四个条件都满足，则将当前image和label存储到新数据集位置
        file_name = os.path.basename(image_path)  # 获取文件名
        # 转移存储
        shutil.copyfile(image_path, os.path.join(new_dataset_image_dir, file_name))
        shutil.copyfile(virtual_label_path, os.path.join(new_dataset_label_dir, file_name))



if __name__ == '__main__':
    # analyse_dataset(dataset_path=r"./datasets/NC-release-data")

    # load_nii_file(r"./datasets/NC-release-data/label/X2313838.nii.gz")

    # load_obj_file(r"./datasets/Teeth3DS/training/upper/0EAKT1CU/0EAKT1CU_upper.obj")

    split_dataset(r"./datasets/NC-release-data-checked", train_ratio=0.8, seed=123)

    # 分析数据集中image和label的一致性和正确性
    # analyse_image_label_consistency(r"./datasets/NC-release-data")



