import os
import re
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

import torch

from thop import profile
from ptflops import get_model_complexity_info
from pytorch_model_summary import summary

import lib.models as models


def analyse_MMOTU_annotations():
    root_dir = r"./datasets/MMOTU/OTU_2d"
    src_dir = os.path.join(root_dir, "annotations")
    dest_dir = os.path.join(root_dir, "processed_annotations")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    # 遍历所有mask图像
    for image_name in os.listdir(src_dir):
        file_name, ext = os.path.splitext(image_name)
        image = Image.open(os.path.join(src_dir, image_name))
        print(np.array(image).shape)
        # 判断是不是binary_binary
        if re.search('binary_binary$', file_name) is not None:
            image = cv2.imread(os.path.join(src_dir, image_name))
            unique_values = np.unique(image)
            if unique_values[1] != 1:
                print("binary_binary: ", unique_values)
            image[image == 1] = 255
            cv2.imwrite(os.path.join(dest_dir, image_name), image)

        elif re.search('binary$', file_name) is not None:
            image = cv2.imread(os.path.join(src_dir, image_name))
            unique_values = np.unique(image)
            if unique_values[1] != 1:
                print("binary: ", unique_values)
            image[image == 1] = 255
            cv2.imwrite(os.path.join(dest_dir, image_name), image)

        else:
            shutil.copyfile(os.path.join(src_dir, image_name), os.path.join(dest_dir, image_name))

    for i in range(1, 1470):
        color_image_name = str(i) + ".PNG"
        binary_image_name = str(i) + "_binary.PNG"
        binary_binary_image_name = str(i) + "_binary_binary.PNG"
        color_image = cv2.imread(os.path.join(src_dir, color_image_name))
        pos_image = np.sum(color_image, axis=2)
        color_image[pos_image > 0] = 1
        binary_image = cv2.imread(os.path.join(src_dir, binary_image_name))
        binary_binary_image = cv2.imread(os.path.join(src_dir, binary_binary_image_name))

        mask1 = (color_image != binary_image)
        mask2 = (binary_image != binary_binary_image)
        if np.sum(mask1) > 0 or np.sum(mask2):
            print(i)


def generate_MMOTU_training_dataset(root_dir):
    src_dir = os.path.join(root_dir, "OTU_2d")
    dest_dir = os.path.join(root_dir, "OTU_2d_processed")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    dest_train_dir = os.path.join(dest_dir, "train")
    os.makedirs(dest_train_dir)
    dest_valid_dir = os.path.join(dest_dir, "valid")
    os.makedirs(dest_valid_dir)
    dest_train_images_dir = os.path.join(dest_train_dir, "images")
    os.makedirs(dest_train_images_dir)
    dest_train_labels_dir = os.path.join(dest_train_dir, "labels")
    os.makedirs(dest_train_labels_dir)
    dest_valid_images_dir = os.path.join(dest_valid_dir, "images")
    os.makedirs(dest_valid_images_dir)
    dest_valid_labels_dir = os.path.join(dest_valid_dir, "labels")
    os.makedirs(dest_valid_labels_dir)
    src_images_dir = os.path.join(src_dir, "images")
    src_annotations_dir = os.path.join(src_dir, "annotations")
    src_train_txt_path = os.path.join(src_dir, "train.txt")
    src_valid_txt_path = os.path.join(src_dir, "val.txt")
    # 依次构造训练子集和验证子集
    with open(src_train_txt_path, "r") as f:
        content = f.readlines()
        for num in content:
            num = num.strip()
            src_image_path = os.path.join(src_images_dir, num + ".JPG")
            dest_image_path = os.path.join(dest_train_images_dir, num + ".JPG")
            shutil.copyfile(src_image_path, dest_image_path)
            src_label_path = os.path.join(src_annotations_dir, num + "_binary_binary.PNG")
            dest_label_path = os.path.join(dest_train_labels_dir, num + ".PNG")
            shutil.copyfile(src_label_path, dest_label_path)
    with open(src_valid_txt_path, "r") as f:
        content = f.readlines()
        for num in content:
            num = num.strip()
            src_image_path = os.path.join(src_images_dir, num + ".JPG")
            dest_image_path = os.path.join(dest_valid_images_dir, num + ".JPG")
            shutil.copyfile(src_image_path, dest_image_path)
            src_label_path = os.path.join(src_annotations_dir, num + "_binary_binary.PNG")
            dest_label_path = os.path.join(dest_valid_labels_dir, num + ".PNG")
            shutil.copyfile(src_label_path, dest_label_path)


def analyse_MMOTU_mean_std(root_dir):
    src_dir = os.path.join(root_dir, "OTU_2d")
    src_images_dir = os.path.join(src_dir, "images")

    sums = np.zeros(3)
    counts = np.zeros(3)
    for image_name in tqdm(os.listdir(src_images_dir)):
        image_path = os.path.join(src_images_dir, image_name)
        image = cv2.imread(image_path)
        image = image / 255
        tmp_sums = image.sum(axis=0).sum(axis=0)
        sums += tmp_sums
        counts += image.shape[0] * image.shape[1]
    means = sums / counts
    print("means:", means)

    std_sums = np.zeros(3)
    for image_name in tqdm(os.listdir(src_images_dir)):
        image_path = os.path.join(src_images_dir, image_name)
        image = cv2.imread(image_path)
        image = image / 255
        tmp_image = (image - means.reshape((1, 1, 3))) ** 2
        tmp_sums = tmp_image.sum(axis=0).sum(axis=0)
        std_sums += tmp_sums
    stds = np.sqrt(std_sums / counts)

    print("stds:", stds)


def cal_MMOTU_weights(root_dir):
    src_dir = os.path.join(root_dir, "OTU_2d_processed")
    src_labels_dir = os.path.join(src_dir, "train", "labels")

    # 初始化统计数组
    statistics_np = np.zeros((2,))
    # 遍历所有图像
    for label_name in tqdm(os.listdir(src_labels_dir)):
        label_path = os.path.join(src_labels_dir, label_name)
        # 读取当前图像的标注图像
        label_np = cv2.imread(label_path)
        # 统计在当前标注图像中出现的类别索引以及各类别索引出现的次数
        class_indexes, indexes_cnt = np.unique(label_np, return_counts=True)
        # 遍历更新到统计数组中
        for j, class_index in enumerate(class_indexes):
            # 获取当前类别索引的次数
            index_cnt = indexes_cnt[j]
            # 累加当前类别索引的次数
            statistics_np[class_index] += index_cnt

    # 初始化权重向量
    weights = np.zeros((2,))
    # 依次计算每个类别的权重
    for i, cnt in enumerate(statistics_np):
        if cnt != 0:
            weights[i] = 1 / cnt
    # 归一化权重数组
    weights = weights / weights.sum()
    print("各类别的权重数组为：", end='[')
    weights_str = ", ".join([str(weight) for weight in weights])
    print(weights_str + "]")


def cal_max_valid_IoU(txtfile_path):
    # 打开文件并读取内容
    with open(txtfile_path, 'r') as file:
        lines = file.readlines()

    # 初始化一个变量来存储最大的 IoU 值
    max_iou = 0.0

    # 遍历每一行，查找 "valid_IoU:" 并提取后面的数字
    for line in lines:
        if 'valid_IoU:' in line:
            # 找到 "valid_IoU:" 的位置
            iou_start = line.find('valid_IoU:')
            if iou_start != -1:
                # 提取 "valid_IoU:" 后面的数字
                iou_str = line[iou_start + len('valid_IoU:'): iou_start + len('valid_IoU:') + 8].strip()
                try:
                    iou_value = float(iou_str)
                    max_iou = max(max_iou, iou_value)
                except ValueError:
                    pass

    # 打印最大的 IoU 值
    print(f"最大的 IoU 值: {max_iou}")


def count_parameters(model):
    """计算PyTorch模型的参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyse_models(model_names_list):
    # 先构造参数字典
    opt = {
        "in_channels": 3,
        "classes": 2,
        "resize_shape": (224, 224),
        "device": "cuda:0",
    }
    # 遍历统计各个模型参数量
    for model_name in model_names_list:
        if model_name != "PMFSNet":
            continue
        # 获取当前模型
        opt["model_name"] = model_name
        model = models.get_model(opt)

        print("***************************************** model name: {} *****************************************".format(model_name))

        print("params: {:.6f}M".format(count_parameters(model) / 1e6))

        input = torch.randn(1, 3, 224, 224).to(opt["device"])
        flops, params = profile(model, (input,))
        print("flops: {:.6f}G, params: {:.6f}M".format(flops / 1e9, params / 1e6))

        try:
            flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
            print("flops: {:.6f}G, params: {:.6f}M".format(flops / 1e9, params / 1e6))
        except:
            continue

        print(summary(model, input, show_input=False, show_hierarchical=False))


def generate_samples_image(scale=2):
    # 创建整个大图
    image = np.full((970, 725, 3), 255)
    # 依次遍历
    for i in range(4):
        x_img, y_img = i * (224 + 8), 0
        img = cv2.imread(r"./images/MMOTU/" + str(i) + "0" + ".JPG")
        img = cv2.resize(img, (360, 224))
        image[x_img: x_img + 224, y_img: y_img + 360, :] = img

        x_lab, y_lab = i * (224 + 8), 365
        lab = cv2.imread(r"./images/MMOTU/" + str(i) + "1" + ".PNG")
        lab = cv2.resize(lab, (360, 224)) * 255
        image[x_lab: x_lab + 224, y_lab: y_lab + 360, :] = lab

    image = image[:, :, ::-1]
    # 添加文字
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    position1 = (80, 925)
    text1 = "Original image"
    draw.text(position1, text1, font=font, fill=color)

    position2 = (450, 925)
    text2 = "Ground truth"
    draw.text(position2, text2, font=font, fill=color)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/MMOTU/MMOTU_samples.jpg")


def generate_segmented_sample_image(scale=1):
    # 创建整个大图
    image = np.full((976, 980, 3), 255)
    # 依次遍历
    for i in range(4):
        for j in range(3):
            pos_x, pos_y = i * (224 + 10), j * (320 + 10)
            img = cv2.imread(r"./images/MMOTU_segment_result_samples/" + str(i) + "_" + str(j) + ".jpg")
            img = cv2.resize(img, (320, 224))
            image[pos_x: pos_x + 224, pos_y: pos_y + 320, :] = img
    image = image[:, :, ::-1]

    # 添加文字的设置
    texts = ["Image", "Ground Truth", "PMFSNet"]
    positions = [110, 390, 750]

    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    # 遍历添加文字
    for i, text in enumerate(texts):
        position = (positions[i], 931)
        draw.text(position, text, font=font, fill=color)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/MMOTU_segment_result_samples/MMOTU_segmentation.jpg")


if __name__ == '__main__':
    # 分析标注文件数据
    # analyse_MMOTU_annotations()

    # 生成用于训练的数据集结构
    # generate_MMOTU_training_dataset(r"./datasets/MMOTU")

    # 分析MMOTU数据集均值和标准差
    # analyse_MMOTU_mean_std(r"./datasets/MMOTU")

    # 计算MMOTU数据集前景和背景加权权重
    # cal_MMOTU_weights(r"./datasets/MMOTU")

    # 计算日志文件中valid_IoU数值最大值
    # cal_max_valid_IoU(r"./log.txt")

    # 依次计算一组模型的计算量和参数量
    # analyse_models(["PMFSNet", "MobileNetV2", "PSPNet", "DANet", "SegFormer", "UNet", "TransUNet", "BiSeNetV2", "MedT"])

    # 生成MMOTU样本展示图
    # generate_samples_image(scale=2)

    # 生成分割后样本拼接图
    generate_segmented_sample_image(scale=1)
