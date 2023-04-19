import numpy as np
import torch
import nrrd
import os
import json
import re
import scipy
from PIL import Image
from scipy import ndimage
from collections import Counter
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt


import lib.utils as utils




def create_sub_volumes(images_path_list, labels_path_list, samples_train, resample_spacing, clip_lower_bound,
                       clip_upper_bound, crop_size, crop_threshold, sub_volume_dir):
    """
    将数据集中的图像裁剪成子卷，组成新的数据集

    Args:
        images_path_list: 所有原图像路径
        labels_path_list: 所有标注图像路径
        samples_train: 训练集子卷数量
        resample_spacing: 重采样的体素间距
        clip_lower_bound: 灰度值下界
        clip_upper_bound: 灰度值上界
        crop_size: 裁剪尺寸
        crop_threshold: 裁剪阈值
        sub_volume_dir: 子卷存储目录

    Returns:

    """
    # 获取图像数量
    image_num = len(images_path_list)
    assert image_num != 0, "原始数据集为空！"
    assert len(images_path_list) == len(labels_path_list), "原始数据集中原图像数量和标注图像数量不一致！"

    # 先把完整标注图像、完整原图图像读取到内存
    image_tensor_list = []
    label_tensor_list = []
    for i in range(image_num):
        # 加载并预处理原图图像
        image_tensor = load_image_or_label(images_path_list[i], resample_spacing, clip_lower_bound,
                                           clip_upper_bound, type="image")
        image_tensor_list.append(image_tensor)
        # 加载并预处理标注图像
        label_tensor = load_image_or_label(labels_path_list[i], resample_spacing, clip_lower_bound,
                                           clip_upper_bound, type="label")
        label_tensor_list.append(label_tensor)

    # 采样指定数量的子卷
    subvolume_list = []
    for i in range(samples_train):
        print("id:", i)
        # 随机对某个图像裁剪子卷
        random_index = np.random.randint(image_num)
        # 获取当前图像数据和标签数据
        image_tensor = image_tensor_list[random_index].clone()
        label_tensor = label_tensor_list[random_index].clone()

        # 反复随机生成裁剪区域，直到满足裁剪指标为止
        cnt_loop = 0
        while True:
            cnt_loop += 1
            # 计算裁剪的位置
            crop_point = find_random_crop_dim(label_tensor.shape, crop_size)
            # 判断当前裁剪区域满不满足条件
            if find_non_zero_labels_mask(label_tensor, crop_threshold, crop_size, crop_point):
                # 裁剪
                crop_image_tensor = crop_img(image_tensor, crop_size, crop_point)
                crop_label_tensor = crop_img(label_tensor, crop_size, crop_point)
                print("loop cnt:", cnt_loop, '\n')
                break

        # 定义子卷原图像和标注图像的公共前缀名
        filename = 'id_' + str(i).zfill(4) + '-src_id_' + str(random_index).zfill(3)
        # 存储子卷原图像
        image_filename = filename + ".npy"
        image_path = os.path.join(sub_volume_dir, "images", image_filename)
        np.save(image_path, crop_image_tensor)
        # 存储子卷标注图像
        label_filename = filename + "_seg.npy"
        label_path = os.path.join(sub_volume_dir, "labels", label_filename)
        np.save(label_path, crop_label_tensor)
        # 记录子卷的路径
        subvolume_list.append((image_path, label_path))

    return subvolume_list



def preprocess_val_dataset(images_path_list, labels_path_list, resample_spacing, clip_lower_bound, clip_upper_bound,
                           sub_volume_dir):
    """
       预处理验证集图像数据

       Args:
           images_path_list: 所有原图像路径
           labels_path_list: 所有标注图像路径
           resample_spacing: 重采样的体素间距
           clip_lower_bound: 灰度值下界
           clip_upper_bound: 灰度值上界
           sub_volume_dir: 子卷存储目录

       Returns:
    """
    # 获取图像数量
    image_num = len(images_path_list)
    assert image_num != 0, "原始数据集为空！"
    assert len(images_path_list) == len(labels_path_list), "原始数据集中原图像数量和标注图像数量不一致！"

    subvolume_list = []
    # 循环读取所有图像
    for i in range(image_num):
        # 提取原图像文件名
        image_file_name = os.path.splitext(os.path.basename(images_path_list[i]))[0]
        # 提取标注图像文件名
        label_file_name = os.path.splitext(os.path.basename(labels_path_list[i]))[0]
        # 原图像文件名要和标注图像文件名一致
        assert image_file_name == label_file_name, "原图像文件名和标注图像文件名不一样！"

        # 加载并预处理原图图像
        image_tensor = load_image_or_label(images_path_list[i], resample_spacing, clip_lower_bound,
                                           clip_upper_bound, type="image")
        # 加载并预处理标注图像
        label_tensor = load_image_or_label(labels_path_list[i], resample_spacing, clip_lower_bound,
                                           clip_upper_bound, type="label")

        # 存储原图图像
        image_filename = image_file_name + ".npy"
        image_path = os.path.join(sub_volume_dir, "images", image_filename)
        np.save(image_path, image_tensor)
        # 存储标注图像
        label_filename = label_file_name + "_seg.npy"
        label_path = os.path.join(sub_volume_dir, "labels", label_filename)
        np.save(label_path, label_tensor)
        # 记录子卷的路径
        subvolume_list.append((image_path, label_path))

    return subvolume_list






def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = crop_size[0]
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = crop_size[1]
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = crop_size[2]
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)





def find_non_zero_labels_mask(label_map, th_percent, crop_size, crop_point):
    segmentation_map = label_map.clone()
    d1, d2, d3 = segmentation_map.shape
    segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = segmentation_map.sum()

    cropped_segm_map = crop_img(segmentation_map, crop_size, crop_point)
    crop_voxel_labels = cropped_segm_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels
    # print(label_percentage,total_voxel_labels,crop_voxel_labels)
    if label_percentage >= th_percent:
        return True
    else:
        return False




def load_image_or_label(path, resample_spacing, clip_lower_bound, clip_upper_bound, type=None):
    """
    加载完整标注图像、随机裁剪后的牙齿图像或标注图像

    Args:
        path: 原始图像路径
        resample_spacing: 重采样的体素间距
        clip_lower_bound: 灰度值下界
        clip_upper_bound: 灰度值上界
        type: 原始图像、原图像标注图像

    Returns:

    """
    # 判断是读取标注文件还是原图像文件
    if type == "label":
        img_np, spacing = load_label(path)
    else:
        img_np, spacing = load_image(path)

    # 定义插值算法
    order = 0 if type == "label" else 3
    # 重采样
    img_np = resample_image_spacing(img_np, spacing, resample_spacing, order)

    # 直接返回tensor
    if type == "label" or type == "ori_image":
        return torch.from_numpy(img_np)

    # 数值上下界clip
    img_np = absolute_value_clip(img_np, min_val=clip_lower_bound, max_val=clip_upper_bound)

    # 转换成tensor
    img_tensor = torch.from_numpy(img_np)

    # 最小灰度值移动到0
    img_tensor -= clip_lower_bound
    # img_tensor -= img_tensor.min()

    return img_tensor





def load_label(path):
    # print(path)
    """
    读取label文件
    Args:
        path: 文件路径

    Returns:

    """
    # 读入 nrrd 文件
    data, options = nrrd.read(path)
    assert data.ndim == 3, "label图像维度出错"
    # 修改数据类型
    data = data.astype(int)

    # 初始化标记字典
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(r"./3DTooth.json")
    class_to_index_dict = {}
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key
    segment_dict = class_to_index_dict.copy()
    for key in segment_dict.keys():
        segment_dict[key] = {"index": int(segment_dict[key]), "color": None, "labelValue": None}

    for key, val in options.items():
        searchObj = re.search(r'^Segment(\d+)_Name$', key)
        if searchObj is not None:
            segment_id = searchObj.group(1)
            # 获取颜色
            segment_color_key = "Segment" + str(segment_id) + "_Color"
            color = options.get(segment_color_key, None)
            if color is not None:
                tmpColor = color.split()
                color = [int(255 * float(c)) for c in tmpColor]
            segment_dict[val]["color"] = color
            # 获取标签值
            segment_label_value_key = "Segment" + str(segment_id) + "_LabelValue"
            labelValue = options.get(segment_label_value_key, None)
            if labelValue is not None:
                labelValue = int(labelValue)
            segment_dict[val]["labelValue"] = labelValue
    # 替换标签值
    for key, val in segment_dict.items():
        if val["labelValue"] is not None:
            # print(key, val["labelValue"])
            data[data == val["labelValue"]] = -val["index"]
    data = -data

    # 获取体素间距
    spacing = [v[i] for i, v in enumerate(options["space directions"])]

    return data, spacing



def load_image(path):
    """
    加载图像数据
    Args:
        path:路径

    Returns:

    """
    # 读取
    data, options = nrrd.read(path)
    assert data.ndim == 3, "图像维度出错"
    # 修改数据类型
    data = data.astype(int)
    # 获取体素间距
    spacing = [v[i] for i, v in enumerate(options["space directions"])]

    return data, spacing




def resample_image_spacing(data, old_spacing, new_spacing, order):
    """
    根据体素间距对图像进行重采样
    Args:
        data:图像数据
        old_spacing:原体素间距
        new_spacing:新体素间距

    Returns:

    """
    scale_list = [old / new_spacing[i] for i, old in enumerate(old_spacing)]
    return scipy.ndimage.interpolation.zoom(data, scale_list, order=order)



def absolute_value_clip(img_numpy, min_val, max_val):
    """
    绝对上下界数值clip
    Args:
        img_numpy: 原始图像
        min_val: clip下界数值
        max_val: clip上界数值

    Returns:

    """
    img_numpy[img_numpy < min_val] = min_val
    img_numpy[img_numpy > max_val] = max_val
    return img_numpy





def crop_img(img_tensor, crop_size, crop_point):
    if crop_size[0] == 0:
        return img_tensor
    slices_crop, w_crop, h_crop = crop_point
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_tensor.dim()
    assert inp_img_dim >= 3
    if img_tensor.dim() == 3:
        full_dim1, full_dim2, full_dim3 = img_tensor.shape
    elif img_tensor.dim() == 4:
        _, full_dim1, full_dim2, full_dim3 = img_tensor.shape
        img_tensor = img_tensor[0, ...]

    if full_dim1 == dim1:
        img_tensor = img_tensor[:, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, :,
                     h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_tensor = img_tensor[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2,
                     h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_tensor.unsqueeze(0)

    return img_tensor














if __name__ == '__main__':
    load_label(r"./datasets/src_10/train/labels/12_2.nrrd")









