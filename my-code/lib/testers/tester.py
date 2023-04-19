import os
import math
import cc3d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from functools import reduce
from nibabel.viewers import OrthoSlicer3D

import configs.config as config

from lib import utils
from lib.visualizations.AverageMeterWriter import AverageMeterWriter
from lib.visualizations.SegmentMapDisplayer import display_compare_hist, display_segmentation_3D



class Tester():
    """
    Tester class
    """
    def __init__(self, model, meter, device):
        """
        Args:
            model: 网络模型
            meter: 评价指标度量器
            device: 设备
        """
        # 传入的参数
        self.model = model
        self.meter = meter
        self.device = device

        # 初始化指标统计打印工具
        self.writer = AverageMeterWriter()


    def split_test(self, image):
        # 获取图像尺寸
        ori_shape = image.size()[2:]
        # 初始化输出的特征图
        output = torch.zeros((image.size()[0], config.classes, *ori_shape), device=self.device)
        # 切片的大小
        slice_shape = config.crop_size
        # 在三个维度上滑动的步长
        stride = config.crop_stride
        # 计算总共要预测多少张切片
        total_slice_num = 1
        for i in range(3):
            total_slice_num *= math.ceil((ori_shape[i] - slice_shape[i]) / stride[i]) + 1

        # 设置进度条
        with tqdm(desc="滑动切片分割", leave=True, total=total_slice_num, unit="slice", ncols=200, ascii=True) as bar:
            # 在三个维度上进行滑动切片
            for shape0_start in range(0, ori_shape[0], stride[0]):
                shape0_end = shape0_start + slice_shape[0]
                start0 = shape0_start
                end0 = shape0_end
                if shape0_end >= ori_shape[0]:
                    end0 = ori_shape[0]
                    start0 = end0 - slice_shape[0]

                for shape1_start in range(0, ori_shape[1], stride[1]):
                    shape1_end = shape1_start + slice_shape[1]
                    start1 = shape1_start
                    end1 = shape1_end
                    if shape1_end >= ori_shape[1]:
                        end1 = ori_shape[1]
                        start1 = end1 - slice_shape[1]

                    for shape2_start in range(0, ori_shape[2], stride[2]):
                        shape2_end = shape2_start + slice_shape[2]
                        start2 = shape2_start
                        end2 = shape2_end
                        if shape2_end >= ori_shape[2]:
                            end2 = ori_shape[2]
                            start2 = end2 - slice_shape[2]

                        slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
                        slice_predict = self.model(slice_tensor.to(self.device))
                        # slice_predict = nn.Softmax(dim=1)(slice_predict)
                        output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict
                        # 进度加1
                        bar.update(1)

                        if shape2_end >= ori_shape[2]:
                            break

                    if shape1_end >= ori_shape[1]:
                        break

                if shape0_end >= ori_shape[0]:
                    break

        return output



    def test_single_image_without_label(self, image):
        self.model.eval()

        with torch.no_grad():
            # 维度扩张
            image = torch.FloatTensor(image.numpy()).unsqueeze(0).unsqueeze(0)
            # 将图像放到GPU上
            image = image.to(self.device)
            # 输入到网络
            output = self.split_test(image)
            # 对各通道进行softmax
            probability_map = nn.Softmax(dim=1)(output)
            # 根据各通道的概率，选择最大值所在的索引作为该处的类别
            class_map = torch.argmax(probability_map, dim=1).squeeze(0).cpu().numpy()
            # 显示
            OrthoSlicer3D(class_map).show()


    def test_single_image(self, image, label):
        self.writer.reset()
        self.model.eval()

        with torch.no_grad():
            # 维度扩张
            image = torch.FloatTensor(image.numpy()).unsqueeze(0).unsqueeze(0)
            label = torch.FloatTensor(label.numpy()).unsqueeze(0)
            # 将图像放到GPU上
            image = image.to(self.device)
            label = label.to(self.device)
            # 输入到网络
            output = self.split_test(image)
            # 计算各通道的dsc
            per_channel_dice = self.meter.get_dice(output, label)
            # 统计并打印
            self.writer.update(per_channel_dice)
            self.writer.display_terminal()
            # 对各通道进行softmax
            probability_map = nn.Softmax(dim=1)(output)
            # 将训练样本中没有的类别概率置为0
            probability_map[:, [26, 34], ...] = 0
            # 根据各通道的概率，选择最大值所在的索引作为该处的类别
            class_map = torch.argmax(probability_map, dim=1).squeeze(0).cpu().numpy()
            # 转换格式
            label_np = label.squeeze(0).cpu().numpy()

            # # 输出概率图
            # print(output[0, :, class_map == 26].permute(1, 0)[:20, :])
            # print(output[0, :, class_map == 26].permute(1, 0)[:20, 26].reshape((-1, 1)))

            # 对预测分割图进行最大连通区域处理
            class_map = maximum_connected_component_analysis(class_map)

            # 显示分布对比图
            display_compare_hist(label_np, class_map)

            # 分割图3D可视化
            display_segmentation_3D(class_map)

            OrthoSlicer3D(label_np).show()
            OrthoSlicer3D(class_map).show()


    def test_image_set(self, test_loader):
        self.writer.reset()
        self.model.eval()

        for image, label in tqdm(test_loader):
            with torch.no_grad():
                # 将图像放到GPU上
                image = image.to(self.device)
                label = label.to(self.device)
                # 输入到网络
                output = self.split_test(image)
                # 计算各通道的dsc
                per_channel_dice = self.meter.get_dice(output, label)
                # 统计并打印
                self.writer.update(per_channel_dice)
        self.writer.display_terminal()



# 筛选每个类别的最大连通区域
def maximum_connected_component_analysis(class_map):
    # 每个类别只保留最大的连通区域，定义列表表示每个类别是否已经处理过了
    rec_label = np.zeros((config.classes, ))
    # 计算总元素个数
    element_num = reduce(lambda x, y: x * y, class_map.shape)
    # 拷贝一份原始图像
    segment_map = class_map.copy()

    # 定义连通性强度
    connectivity = 26
    # 进行连通区域分析标记，返回标记三维数组
    labels_out, N = cc3d.connected_components(segment_map, return_N=True, connectivity=connectivity)
    # 进一步获取连通区域的统计信息
    """
    {
      voxel_counts: np.ndarray[uint64_t] (index is label) (N+1)

      # Structure is xmin,xmax,ymin,ymax,zmin,zmax by label
      bounding_boxes: np.ndarray[uint64_t] (N+1 x 6)

      # Structure is x,y,z
      centroids: np.ndarray[float64] (N+1,3)
    }
    """
    statistics_data = cc3d.statistics(labels_out)
    # 得到每个连通区域的大小，索引表示连通区域的编号，元素值为该连通区域大小
    voxel_counts = statistics_data["voxel_counts"]
    # 构造所有连通区域面积和label构成的元组的列表
    components = [(voxel_counts[i], i) for i in range(1, N+1)]
    # 按照连通区域面积从大到小排序
    components = sorted(components, key=lambda x: x[0], reverse=True)

    # 遍历所有的连通区域
    for i in range(N):
        # 获取当前连通区域的label
        ccl_label = components[i][1]
        # 获取当前连通区域的原数据值
        ori_label = class_map[labels_out == ccl_label][0]
        # 判断当前得到的原数值代表的牙齿类别是否已经处理过
        if (ori_label == 1 and components[i][0] < 0.0001 * element_num) or (ori_label != 1 and rec_label[ori_label] == 1):
            # 该连通区域不是该牙齿类别的最大连通区域, 设置为背景0
            segment_map[labels_out == ccl_label] = 0
        elif ori_label != 1:
            # 该连通区域是该牙齿类别的最大连通区域
            rec_label[ori_label] = 1
    return segment_map















