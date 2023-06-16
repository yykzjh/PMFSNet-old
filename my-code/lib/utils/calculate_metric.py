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


def compute_per_channel_hd(seg, target, num_classes, c=6):
    """
    计算各类别的hd
    :param seg: 预测分割后的分割图
    :param target: 真实标签图
    :param num_classes: 通道和类别数
    :param c: 连通度
    :return:
    """
    if c == 6:
        neighbors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif c == 18:
        neighbors = [[-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1],
                     [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, 0], [1, 0, -1], [1, 0, 0],
                     [1, 0, 1], [1, 1, 0]]
    elif c == 26:
        neighbors = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1],
                     [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1],
                     [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1],
                     [1, 1, -1], [1, 1, 0], [1, 1, 1]]
    else:
        raise RuntimeError(f"不支持连通度为{c}")

    # 获取各维度大小
    bs, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算HD
    for b in range(bs):  # 遍历batch
        # 初始化存储各个类别的表面点坐标的数据结构
        seg_surface = [[] for _ in range(num_classes)]
        target_surface = [[] for _ in range(num_classes)]
        # 遍历seg图像
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    # 获取当前类别索引
                    cur_class = seg[b, i, j, k]
                    # 遍历周围的点
                    for di, dj, dk in neighbors:
                        # 获得周围的点坐标
                        tmpi = i + di
                        tmpj = j + dj
                        tmpk = k + dk
                        # 周围的点与当前点的类别索引不相同
                        if 0 <= tmpi < h and 0 <= tmpj < w and 0 <= tmpk < d and seg[b, tmpi, tmpj, tmpk] == cur_class:
                            continue
                        # 记录表面点
                        seg_surface[cur_class].append([i, j, k])
                        break
        # 遍历target图像
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    # 获取当前类别索引
                    cur_class = target[b, i, j, k]
                    # 遍历周围的点
                    for di, dj, dk in neighbors:
                        # 获得周围的点坐标
                        tmpi = i + di
                        tmpj = j + dj
                        tmpk = k + dk
                        # 周围的点与当前点的类别索引不相同
                        if 0 <= tmpi < h and 0 <= tmpj < w and 0 <= tmpk < d and target[b, tmpi, tmpj, tmpk] == cur_class:
                            continue
                        # 记录表面点
                        target_surface[cur_class].append([i, j, k])
                        break
        # 遍历各类别
        for cla in range(num_classes):
            if len(seg_surface[cla]) == 0 or len(target_surface[cla]) == 0:
                continue
            # 计算seg到target的HD
            HD_max_val1 = 0
            for seg_point in seg_surface[cla]:
                HD_min_val = math.sqrt(d * d + h * h + w * w)
                for target_point in target_surface[cla]:
                    l2_distance = np.linalg.norm(np.array(seg_point) - np.array(target_point), ord=2, keepdims=False)
                    HD_min_val = min(HD_min_val, l2_distance)
                HD_max_val1 = max(HD_max_val1, HD_min_val)
            # 计算target到seg的HD
            HD_max_val2 = 0
            for target_point in target_surface[cla]:
                HD_min_val = math.sqrt(d * d + h * h + w * w)
                for seg_point in seg_surface[cla]:
                    l2_distance = np.linalg.norm(np.array(target_point) - np.array(seg_point), ord=2, keepdims=False)
                    HD_min_val = min(HD_min_val, l2_distance)
                HD_max_val2 = max(HD_max_val2, HD_min_val)
            # 计算当前类别的HD
            HD_per_class = max(HD_max_val1, HD_max_val2)
            # 添加到结果数组
            output[b, cla] = HD_per_class
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


def compute_per_channel_hd_lib(seg, target, num_classes):
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


def compute_per_channel_assd(seg, target, num_classes, c=6):
    """
    计算各类别的assd
    :param seg: 分割后的分割图
    :param target: 真实标签图
    :param num_classes: 通道和类别数
    :param c: 连通度
    :return:
    """
    if c == 6:
        neighbors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif c == 18:
        neighbors = [[-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1],
                     [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, 0], [1, 0, -1], [1, 0, 0],
                     [1, 0, 1], [1, 1, 0]]
    elif c == 26:
        neighbors = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1],
                     [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1],
                     [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1],
                     [1, 1, -1], [1, 1, 0], [1, 1, 1]]
    else:
        raise RuntimeError(f"不支持连通度为{c}")

    # 获取各维度大小
    bs, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算ASSD
    for b in range(bs):  # 遍历batch
        # 初始化存储各个类别的表面点坐标的数据结构
        seg_surface = [[] for _ in range(num_classes)]
        target_surface = [[] for _ in range(num_classes)]
        # 遍历seg图像
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    # 获取当前类别索引
                    cur_class = seg[b, i, j, k]
                    # 遍历周围的点
                    for di, dj, dk in neighbors:
                        # 获得周围的点坐标
                        tmpi = i + di
                        tmpj = j + dj
                        tmpk = k + dk
                        # 周围的点与当前点的类别索引不相同
                        if 0 <= tmpi < h and 0 <= tmpj < w and 0 <= tmpk < d and seg[b, tmpi, tmpj, tmpk] == cur_class:
                            continue
                        # 记录表面点
                        seg_surface[cur_class].append([i, j, k])
                        break
        # 遍历target图像
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    # 获取当前类别索引
                    cur_class = target[b, i, j, k]
                    # 遍历周围的点
                    for di, dj, dk in neighbors:
                        # 获得周围的点坐标
                        tmpi = i + di
                        tmpj = j + dj
                        tmpk = k + dk
                        # 周围的点与当前点的类别索引不相同
                        if 0 <= tmpi < h and 0 <= tmpj < w and 0 <= tmpk < d and target[b, tmpi, tmpj, tmpk] == cur_class:
                            continue
                        # 记录表面点
                        target_surface[cur_class].append([i, j, k])
                        break
        # 遍历各类别
        for cla in range(num_classes):
            print(len(seg_surface[cla]), len(target_surface[cla]))
            if len(seg_surface[cla]) == 0 or len(target_surface[cla]) == 0:
                continue
            # 计算seg到target的ASSD
            ASSD_sum1 = 0
            for seg_point in seg_surface[cla]:
                ASSD_min_val = math.sqrt(d * d + h * h + w * w)
                for target_point in target_surface[cla]:
                    l2_distance = np.linalg.norm(np.array(seg_point) - np.array(target_point), ord=2, keepdims=False)
                    ASSD_min_val = min(ASSD_min_val, l2_distance)
                ASSD_sum1 += ASSD_min_val
            # 计算target到seg的ASSD
            ASSD_sum2 = 0
            for target_point in target_surface[cla]:
                ASSD_min_val = math.sqrt(d * d + h * h + w * w)
                for seg_point in seg_surface[cla]:
                    l2_distance = np.linalg.norm(np.array(target_point) - np.array(seg_point), ord=2, keepdims=False)
                    ASSD_min_val = min(ASSD_min_val, l2_distance)
                ASSD_sum2 += ASSD_min_val
            # 计算当前类别的ASSD
            ASSD_per_class = (ASSD_sum1 + ASSD_sum2) / (len(seg_surface[cla]) + len(target_surface[cla]))
            # 添加到结果数组
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


def compute_per_channel_assd_lib(seg, target, num_classes):
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
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(),
                                                             spacing_mm=(1.0, 1.0, 1.0))
            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            # 计算一张图像一个类别的assd
            assd_tuple = sd.compute_average_surface_distance(surface_distances)
            print(len(surface_distances["distances_pred_to_gt"]), len(surface_distances["distances_gt_to_pred"]))

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


def compute_per_channel_so(seg, target, num_classes, c=6, theta=1.0):
    """
    计算各类别的so

    :param seg: 分割后的分割图
    :param target: 真实标签图
    :param num_classes: 通道和类别数
    :param c: 连通度
    :param theta: 判断两个点处于相同位置的最大距离
    :return:
    """
    if c == 6:
        neighbors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif c == 18:
        neighbors = [[-1, -1, 0], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [0, -1, -1], [0, -1, 0], [0, -1, 1],
                     [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, 0], [1, 0, -1], [1, 0, 0],
                     [1, 0, 1], [1, 1, 0]]
    elif c == 26:
        neighbors = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1],
                     [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1],
                     [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1],
                     [1, 1, -1], [1, 1, 0], [1, 1, 1]]
    else:
        raise RuntimeError(f"不支持连通度为{c}")

    # 获取各维度大小
    bs, h, w, d = seg.shape
    # 初始化输出
    output = torch.full((bs, num_classes), -1.0)

    # 计算SO
    for b in range(bs):  # 遍历batch
        # 初始化存储各个类别的表面点坐标的数据结构
        seg_surface = [[] for _ in range(num_classes)]
        target_surface = [[] for _ in range(num_classes)]
        # 遍历seg图像
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    # 获取当前类别索引
                    cur_class = seg[b, i, j, k]
                    # 遍历周围的点
                    for di, dj, dk in neighbors:
                        # 获得周围的点坐标
                        tmpi = i + di
                        tmpj = j + dj
                        tmpk = k + dk
                        # 周围的点与当前点的类别索引不相同
                        if 0 <= tmpi < h and 0 <= tmpj < w and 0 <= tmpk < d and seg[b, tmpi, tmpj, tmpk] == cur_class:
                            continue
                        # 记录表面点
                        seg_surface[cur_class].append([i, j, k])
                        break
        # 遍历target图像
        for i in range(h):
            for j in range(w):
                for k in range(d):
                    # 获取当前类别索引
                    cur_class = target[b, i, j, k]
                    # 遍历周围的点
                    for di, dj, dk in neighbors:
                        # 获得周围的点坐标
                        tmpi = i + di
                        tmpj = j + dj
                        tmpk = k + dk
                        # 周围的点与当前点的类别索引不相同
                        if 0 <= tmpi < h and 0 <= tmpj < w and 0 <= tmpk < d and target[b, tmpi, tmpj, tmpk] == cur_class:
                            continue
                        # 记录表面点
                        target_surface[cur_class].append([i, j, k])
                        break
        # 遍历各类别
        for cla in range(num_classes):
            if len(seg_surface[cla]) == 0 or len(target_surface[cla]) == 0:
                continue
            # 计算seg中每个点到target表面点集合的距离
            SO_sum = 0
            for seg_point in seg_surface[cla]:
                SO_min_val = math.sqrt(d * d + h * h + w * w)
                for target_point in target_surface[cla]:
                    l2_distance = np.linalg.norm(np.array(seg_point) - np.array(target_point), ord=2, keepdims=False)
                    SO_min_val = min(SO_min_val, l2_distance)
                if SO_min_val < theta:
                    SO_sum += 1

            # 计算当前类别的SO
            SO_per_class = SO_sum / len(seg_surface[cla])
            # 添加到结果数组
            output[b, cla] = SO_per_class
    # 计算各类别在batch上的平均SO
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





