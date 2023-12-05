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
import torch
from torchinfo import summary
from thop import profile
from ptflops import get_model_complexity_info
from torchstat import stat
import nibabel as nib
from tqdm import tqdm
from functools import reduce
from nibabel.viewers import OrthoSlicer3D
import SimpleITK as sitk
from proplot import rc
import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageDraw, ImageFont

import lib.utils as utils
import lib.models as models




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


def analyse_image_label_consistency(dataset_dir, resample_spacing=(0.5, 0.5, 0.5)):
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
        image_np = utils.load_image_or_label(image_path, resample_spacing, type="image")
        label_np = utils.load_image_or_label(virtual_label_path, resample_spacing, type="label")
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


def analyse_dataset(dataset_dir, resample_spacing=(0.5, 0.5, 0.5), clip_lower_bound_ratio=0.01, clip_upper_bound_ratio=0.995, classes=2):
    # 加载数据集中所有原图像
    images_list = sorted(glob.glob(os.path.join(dataset_dir, "image", "*.nii.gz")))
    # 加载数据集中所有标注图像
    labels_list = sorted(glob.glob(os.path.join(dataset_dir, "label", "*.nii.gz")))
    assert len(images_list) == len(labels_list), "原图像数量和标注图像数量不一致"

    print("开始统计体素总量、下界百分位到最小值的长度、上界百分位到最大值的长度->")
    # 先统计数据集的总体素量
    total_voxels = 0
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像和标注图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        # 累计体素
        total_voxels += reduce(lambda acc, cur: acc * cur, image_np.shape)
    # 计算上下界分位点需要存储的数据量
    lower_length = int(total_voxels * clip_lower_bound_ratio + 1)
    upper_length = int(total_voxels - total_voxels * clip_upper_bound_ratio + 1)
    print("体素总量:{}, 下界百分位到最小值的长度:{}, 上界百分位到最大值的长度:{}".format(total_voxels, lower_length, upper_length))

    print("开始计算所有元素和前景元素的最小值和最大值、clip的上下界->")
    # 初始化数据结构
    lower_all_values = np.array([])
    upper_all_values = np.array([])
    all_values_min = 1e9
    all_values_max = -1e9
    foreground_values_min = 1e9
    foreground_values_max = -1e9
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像和标注图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        label_np = utils.load_image_or_label(labels_list[i], resample_spacing, type="label")
        # 将当前图像的体素添加到上下界的存储数组中并排序
        lower_all_values = np.sort(np.concatenate([lower_all_values, image_np.flatten()], axis=0))
        upper_all_values = np.concatenate([upper_all_values, image_np.flatten()], axis=0)
        upper_all_values = upper_all_values[np.argsort(-upper_all_values)]
        # 判断需不需要裁剪所需的部分
        if len(lower_all_values) > lower_length:
            lower_all_values = lower_all_values[:lower_length]
        if len(upper_all_values) > upper_length:
            upper_all_values = upper_all_values[:upper_length]
        # 维护全部体素和前景体素的最小值和最大值
        all_values_min = min(all_values_min, image_np.min())
        all_values_max = max(all_values_max, image_np.max())
        foreground_values_min = min(foreground_values_min, image_np[label_np != 0].min())
        foreground_values_max = max(foreground_values_max, image_np[label_np != 0].max())

    # 计算指定的上下界分位点的灰度值
    clip_lower_bound = lower_all_values[lower_length - 1]
    clip_upper_bound = upper_all_values[upper_length - 1]

    # 输出所有元素和前景元素的最小值和最大值
    print("all_values_min:{}, fore_values_min:{}, fore_values_max:{}, all_values_max:{}"
          .format(all_values_min, foreground_values_min, foreground_values_max, all_values_max))
    # 输出clip的上下界
    print("clip_lower_bound:{}, clip_upper_bound:{}".format(clip_lower_bound, clip_upper_bound))


    print("开始计算均值和方差->")
    # 初始化均值的加和
    mean_sum = 0
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        # 对所有灰度值进行clip
        image_np[image_np < clip_lower_bound] = clip_lower_bound
        image_np[image_np > clip_upper_bound] = clip_upper_bound
        # 先将当前图像进行归一化
        image_np = (image_np - clip_lower_bound) / (clip_upper_bound - clip_lower_bound)
        # 累加
        mean_sum += np.sum(image_np)
    # 计算均值
    mean = mean_sum / total_voxels

    # 初始化标准差的加和
    std_sum = 0
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        # 对所有灰度值进行clip
        image_np[image_np < clip_lower_bound] = clip_lower_bound
        image_np[image_np > clip_upper_bound] = clip_upper_bound
        # 先将当前图像进行归一化
        image_np = (image_np - clip_lower_bound) / (clip_upper_bound - clip_lower_bound)
        # 计算当前图像每个灰度值减去均值的平方
        image_np = (image_np - mean) ** 2
        # 累加
        std_sum += np.sum(image_np)
    # 计算标准差
    std = np.sqrt(std_sum / total_voxels)

    print("均值为:{}, 标准差为:{}".format(mean, std))


    print("开始计算每个类别的权重数组->")
    # 初始化统计数组
    statistics_np = np.zeros((classes, ))
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的标注图像
        label_np = utils.load_image_or_label(labels_list[i], resample_spacing, type="label")
        # 统计在当前标注图像中出现的类别索引以及各类别索引出现的次数
        class_indexes, indexes_cnt = np.unique(label_np, return_counts=True)
        # 遍历更新到统计数组中
        for j, class_index in enumerate(class_indexes):
            # 获取当前类别索引的次数
            index_cnt = indexes_cnt[j]
            # 累加当前类别索引的次数
            statistics_np[class_index] += index_cnt

    # 初始化权重向量
    weights = np.zeros((classes, ))
    # 依次计算每个类别的权重
    for i, cnt in enumerate(statistics_np):
        if cnt != 0:
            weights[i] = 1 / cnt
    # 归一化权重数组
    weights = weights / weights.sum()
    print("各类别的权重数组为：", end='[')
    weights_str = ", ".join([str(weight) for weight in weights])
    print(weights_str + "]")


def count_parameters(model):
    """计算PyTorch模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def count_all_models_parameters(model_names_list):
    # 先构造参数字典
    opt = {
        "in_channels": 1,
        "classes": 2,
        "device": "cpu",
    }
    # 遍历统计各个模型参数量
    for model_name in model_names_list:
        if model_name != "PMRFNet":
            continue
        # 获取当前模型
        opt["model_name"] = model_name
        model = models.get_model(opt)

        print("***************************************** model name: {} *****************************************".format(model_name))

        print("params: {:.6f}M".format(count_parameters(model)))

        input = torch.randn(1, 1, 160, 160, 96).to(opt["device"])
        flops, params = profile(model, (input,))
        print("flops: {:.6f}G, params: {:.6f}M".format(flops / 1e9, params / 1e6))

        flops, params = get_model_complexity_info(model, (1, 160, 160, 96), as_strings=False, print_per_layer_stat=False)
        print("flops: {:.6f}G, params: {:.6f}M".format(flops / 1e9, params / 1e6))


def generate_NC_release_data_snapshot(root_dir, size=224):
    src_image_dir = os.path.join(root_dir, "NC-release-data", "image")
    src_label_dir = os.path.join(root_dir, "NC-release-data", "label")
    dst_root_dir = os.path.join(root_dir, "snapshots")
    if os.path.exists(dst_root_dir):
        shutil.rmtree(dst_root_dir)
    os.makedirs(dst_root_dir)

    # 加载数据集中所有原图像
    images_list = sorted(glob.glob(os.path.join(src_image_dir, "*.nii.gz")))
    labels_list = sorted(glob.glob(os.path.join(src_label_dir, "*.nii.gz")))
    for i in tqdm(range(len(images_list))):
        file_name = os.path.splitext(os.path.splitext(os.path.basename(images_list[i]))[0])[0]
        image_dir = os.path.join(dst_root_dir, file_name)
        os.makedirs(image_dir)
        # image_np = utils.load_image_or_label(images_list[i], (0.5, 0.5, 0.5), type="image")
        label_np = utils.load_image_or_label(labels_list[i], (0.5, 0.5, 0.5), type="label")
        h, w, d = label_np.shape
        s = (d - 30) // 2
        for j in range(6):
            image_snapshot = label_np[:, :, s + 5 * j]
            image_snapshot = cv2.transpose(image_snapshot)
            image_snapshot = cv2.flip(image_snapshot, 0)
            # image_snapshot = np.clip(image_snapshot, -450, 1450)
            # image_snapshot = cv2.resize(image_snapshot, (size, size))
            # cv2.imwrite(os.path.join(image_dir, str(j) + ".jpg"), image_snapshot)
            plt.imshow(image_snapshot, cmap="gray")
            plt.show()


def compare_Dice():
    # 统一设置字体
    rc["font.family"] = "Times New Roman"
    # 统一设置轴刻度标签的字体大小
    rc['tick.labelsize'] = 14
    # 统一设置xy轴名称的字体大小
    rc["axes.labelsize"] = 16
    # 统一设置轴刻度标签的字体粗细
    rc["axes.labelweight"] = "light"
    # 统一设置xy轴名称的字体粗细
    rc["tick.labelweight"] = "light"

    x = np.linspace(0, 1, 1000)
    y1 = 1 - ((2 * x) / (x + 1))
    y2 = 1 - ((2 * x) / (x**2 + 1))

    plt.plot(x, y1, label="Standard Dice Loss")
    plt.plot(x, y2, label="Extended Dice Loss")
    plt.legend(fontsize=14)
    plt.xlabel("probability of ground truth class")
    plt.ylabel("loss")


    plt.savefig('loss.jpg', dpi=600, bbox_inches='tight')


def generate_samples_image(scale=2):
    # 创建整个大图
    image = np.full((970, 1090, 3), 255)
    # 依次遍历
    for i in range(4):
        for j in range(3):
            x = i * (224 + 8)
            y = j * (360 + 5)
            # 读取并处理图像
            img = cv2.imread(r"./images/" + str(i) + str(j) + ".jpg")
            img = cv2.resize(img, (360, 224))
            image[x: x + 224, y: y + 360, :] = img
    image = image[:, :, ::-1]
    # 添加文字
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    position1 = (80, 925)
    text1 = "Original image"
    draw.text(position1, text1, font=font, fill=color)

    position2 = (410, 925)
    text2 = "Ground truth (2D)"
    draw.text(position2, text2, font=font, fill=color)

    position3 = (780, 925)
    text3 = "Ground truth (3D)"
    draw.text(position3, text3, font=font, fill=color)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/3D_CBCT_Tooth_samples.jpg")





if __name__ == '__main__':

    # load_nii_file(r"./datasets/NC-release-data/label/X2313838.nii.gz")

    # load_obj_file(r"./datasets/Teeth3DS/training/upper/0EAKT1CU/0EAKT1CU_upper.obj")

    # split_dataset(r"./datasets/NC-release-data-checked", train_ratio=0.8, seed=123)

    # 分析数据集中image和label的一致性和正确性
    # analyse_image_label_consistency(r"./datasets/NC-release-data")

    # 分析数据集的Clip上下界、均值和方差
    # analyse_dataset(dataset_dir=r"./datasets/NC-release-data-checked", resample_spacing=[0.5, 0.5, 0.5], clip_lower_bound_ratio=1e-6, clip_upper_bound_ratio=1-1e-7)

    # 统计所有网络模型的参数量
    count_all_models_parameters(["DenseVNet", "UNet3D", "VNet", "AttentionUNet", "R2UNet", "R2AttentionUNet",
                                 "HighResNet3D", "DenseVoxelNet", "MultiResUNet3D", "DenseASPPUNet", "UNETR", "PMRFNet"])

    # 生成牙齿数据集快照
    # generate_NC_release_data_snapshot(r"./datasets")

    # 对别两类Dice Loss
    # compare_Dice()

    # 生成牙齿数据集的样本展示图
    # generate_samples_image(scale=2)

