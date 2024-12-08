import math
import numpy as np
import torch

import surface_distance as sd


def compute_per_channel_dice(input, target, mode="extension", epsilon=1e-6):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    :param input: 网络输出的经过Sigmoid或者Softmax归一化的预测概率图
    :param target: 真实标签图
    :param mode: DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式
    :param epsilon: 分母最小值，确保除法的稳定性
    :return:
    """

    # 判断预测图和标注图的尺寸是否一致
    assert input.size() == target.size(), "计算dsc时input和target的尺寸不一致!"

    # 都压缩成二维tensor,(C, B * H * W * D)
    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # 计算DSC的分子
    intersect = (input * target).sum(-1)

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    if mode == "extension":
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
    elif mode == "standard":
        denominator = input.sum(-1) + target.sum(-1)

    # 返回最终计算得到的DSC,(C, )
    return (2 * intersect + epsilon) / (denominator + epsilon)


def compute_per_channel_hd(seg, target, num_classes):
    """
    计算各类别的hd
    :param seg: 预测分割后的分割图
    :param target: 真实标签图
    :param num_classes: 通道和类别数
    :return:
    """
    # 获取各维度大小
    bs, _, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算HD
    for b in range(bs):  # 遍历batch
        # 遍历各类别
        for cla in range(num_classes):
            # 分别计算两个表面点集合中各点到对面集合的距离
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(),
                                                             spacing_mm=(1.0, 1.0, 1.0))
            
            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            # 计算一张图像一个类别的hd
            output[b, cla] = sd.compute_robust_hausdorff(surface_distances, 95)
    # 计算各类别在batch上的平均HD
    out = torch.full((num_classes, ), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_assd(seg, target, num_classes):
    """
    计算各类别的assd
    :param seg: 分割后的分割图
    :param target: 真实标签图
    :param num_classes: 通道和类别数
    :return:
    """
    # 获取各维度大小
    bs, _, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算ASSD
    for b in range(bs):  # 遍历batch
        # 遍历各类别
        for cla in range(num_classes):
            # 分别计算两个表面点集合中各点到对面集合的距离
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(), spacing_mm=(1.0, 1.0, 1.0))
            
            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            # 计算一张图像一个类别的assd
            assd_tuple = sd.compute_average_surface_distance(surface_distances)
            ASSD_per_class = ((assd_tuple[0] * len(surface_distances["distances_gt_to_pred"]) + assd_tuple[1] * len(surface_distances["distances_pred_to_gt"])) /
                              (len(surface_distances["distances_gt_to_pred"]) +
                              len(surface_distances["distances_pred_to_gt"])))
            output[b, cla] = ASSD_per_class
    # 计算各类别在batch上的平均ASSD
    out = torch.full((num_classes, ), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_so(seg, target, num_classes, theta=1.0):
    """
    计算各类别的so

    :param seg: 分割后的分割图
    :param target: 真实标签图
    :param num_classes: 通道和类别数
    :param theta: 判断两个点处于相同位置的最大距离
    :return:
    """
    # 获取各维度大小
    bs, _, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算SO
    for b in range(bs):  # 遍历batch
        # 遍历各类别
        for cla in range(num_classes):
            # 分别计算两个表面点集合中各点到对面集合的距离
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(),
                                                             spacing_mm=(1.0, 1.0, 1.0))
            
            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(
                    surface_distances["distances_gt_to_pred"]) == 0:
                continue
            # 计算一张图像一个类别的so
            so_tuple = sd.compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm=theta)
            output[b, cla] = so_tuple[1]
    # 计算各类别在batch上的平均SO
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_sd(seg, target, num_classes, theta=1.0):
    """
        计算各类别的sd

        :param seg: 分割后的分割图
        :param target: 真实标签图
        :param num_classes: 通道和类别数
        :param theta: 判断两个点处于相同位置的最大距离
        :return:
        """
    # 获取各维度大小
    bs, _, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算SD
    for b in range(bs):  # 遍历batch
        # 遍历各类别
        for cla in range(num_classes):
            # 分别计算两个表面点集合中各点到对面集合的距离
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(),
                                                             spacing_mm=(1.0, 1.0, 1.0))
            
            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            # 计算一张图像一个类别的sd
            sd_score = sd.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=theta)
            output[b, cla] = sd_score
    # 计算各类别在batch上的平均SD
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_iou(seg, target, num_classes):
    """
        计算各类别的sd

        :param seg: 分割后的分割图
        :param target: 真实标签图
        :param num_classes: 通道和类别数
        :return:
        """
    # 获取各维度大小
    bs, _, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算IoU
    for b in range(bs):  # 遍历batch
        # 遍历各类别
        for cla in range(num_classes):
            # 计算两个元素都为1
            intersection = np.logical_and(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy())
            # 计算只要有一个元素为1
            union = np.logical_or(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy())
            iou_score = np.sum(intersection) / np.sum(union)
            output[b, cla] = iou_score
    # 计算各类别在batch上的平均IoU
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W, D) -> (C, N * H * W * D)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)





