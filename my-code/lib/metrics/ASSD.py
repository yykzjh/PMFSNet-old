import math
import torch
import numpy as np



class AverageSymmetricSurfaceDistance(object):
    def __init__(self, c=6, num_classes=33):
        """
        定义平均对称表面距离评价指标计算器
        Args:
            c: 连通度
            num_classes: 类别数
        """
        super(AverageSymmetricSurfaceDistance, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        if c == 6:
            self.neighbors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        elif c == 26:
            self.neighbors = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1],
                              [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1],
                              [0, 1, -1], [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1],
                              [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]
        else:
            raise RuntimeError(f"不支持连通度为{c}")


    def __call__(self, pred, gt):
        """
        计算平均对称表面距离评价指标
        Args:
            pred: 经过softmax后的预测图像
            gt: ground truth 图像

        Returns: 各batch中各类别牙齿的平均豪斯多夫距离
        """
        # 先将预测图像进行分割
        seg = torch.argmax(pred, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == gt.shape, "seg和gt的维度大小不一致"
        # 转换seg和gt数据类型为整型
        seg = seg.type(torch.uint8)
        gt = gt.type(torch.uint8)
        # 获取各维度大小
        bs, d, h, w = seg.shape
        # 初始化输出
        output = torch.zeros((bs, ))

        # 计算ASSD
        for b in range(bs):  # 遍历batch
            # 初始化存储各个类别的表面点坐标的数据结构
            seg_surface = [[] for _ in range(self.num_classes)]
            gt_surface = [[] for _ in range(self.num_classes)]
            # 遍历seg图像
            for i in range(d):
                for j in range(h):
                    for k in range(w):
                        # 获取当前类别索引
                        cur_class = seg[b, i, j, k]
                        if cur_class == 0:  # 当前类别索引不能是0
                            continue
                        # 遍历周围的点
                        for di, dj, dk in self.neighbors:
                            # 获得周围的点坐标
                            tmpi = i + di
                            tmpj = j + dj
                            tmpk = k + dk
                            # 周围的点与当前点的类别索引不相同
                            if 0 <= tmpi < d and 0 <= tmpj < h and 0 <= tmpk < w and seg[b, tmpi, tmpj, tmpk] == cur_class:
                                continue
                            # 记录表面点
                            seg_surface[cur_class].append([i, j, k])
                            break
            # 遍历gt图像
            for i in range(d):
                for j in range(h):
                    for k in range(w):
                        # 获取当前类别索引
                        cur_class = gt[b, i, j, k]
                        if cur_class == 0:  # 当前类别索引不能是0
                            continue
                        # 遍历周围的点
                        for di, dj, dk in self.neighbors:
                            # 获得周围的点坐标
                            tmpi = i + di
                            tmpj = j + dj
                            tmpk = k + dk
                            # 周围的点与当前点的类别索引不相同
                            if 0 <= tmpi < d and 0 <= tmpj < h and 0 <= tmpk < w and gt[b, tmpi, tmpj, tmpk] == cur_class:
                                continue
                            # 记录表面点
                            gt_surface[cur_class].append([i, j, k])
                            break
            # 计算当前图像在预测图像中的表面点集合和真实标签图像中的表面点集合的个类别平均ASSD
            # 定义一些参数
            ASSD_sum = 0
            ASSD_cnt = 0
            # 遍历个类别
            for cla in range(1, self.num_classes):
                if len(seg_surface[cla]) == 0 or len(gt_surface[cla]) == 0:
                    continue
                seg_total = (seg == cla).sum()
                gt_total = (gt == cla).sum()
                print(len(seg_surface[cla]), seg_total, len(gt_surface[cla]), gt_total)
                # 计算seg到gt的ASSD
                ASSD_sum1 = 0
                for seg_point in seg_surface[cla]:
                    ASSD_min_val = math.sqrt(d * d + h * h + w * w)
                    for gt_point in gt_surface[cla]:
                        l2_distance = np.linalg.norm(np.array(seg_point) - np.array(gt_point), ord=2, keepdims=False)
                        ASSD_min_val = min(ASSD_min_val, l2_distance)
                    ASSD_sum1 += ASSD_min_val
                # 计算gt到seg的ASSD
                ASSD_sum2 = 0
                for gt_point in gt_surface[cla]:
                    ASSD_min_val = math.sqrt(d * d + h * h + w * w)
                    for seg_point in seg_surface[cla]:
                        l2_distance = np.linalg.norm(np.array(gt_point) - np.array(seg_point), ord=2, keepdims=False)
                        ASSD_min_val = min(ASSD_min_val, l2_distance)
                    ASSD_sum2 += ASSD_min_val
                # 计算当前类别的ASSD
                ASSD_per_class = (ASSD_sum1 + ASSD_sum2) / (len(seg_surface[cla]) + len(gt_surface[cla]))
                # 累加
                ASSD_sum += ASSD_per_class
                ASSD_cnt += 1
            # 计算一张图像的平均HD
            output[b] = ASSD_sum / ASSD_cnt

        return output



if __name__ == '__main__':
    pred = torch.randn((4, 33, 32, 32, 16))
    gt = torch.randint(33, (4, 32, 32, 16))

    pred = torch.nn.Softmax(dim=1)(pred)

    ASSD_metric = AverageSymmetricSurfaceDistance(c=6, num_classes=33)

    batch_ASSD = ASSD_metric(pred, gt)

    print(batch_ASSD)




















