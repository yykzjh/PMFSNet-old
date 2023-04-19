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




if __name__ == '__main__':
    # analyse_dataset(dataset_path=r"./datasets/NC-release-data")

    # load_nii_file(r"./datasets/NC-release-data/label/X2313838.nii.gz")

    # load_obj_file(r"./datasets/Teeth3DS/training/upper/0EAKT1CU/0EAKT1CU_upper.obj")

    split_dataset(r"./datasets/NC-release-data-modify", train_ratio=0.8, seed=123)


