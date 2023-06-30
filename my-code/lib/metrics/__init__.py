# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:50
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
from .HD import *
from .ASSD import *
from .DICE import *
from .SO import *


def get_metric(opt):
    # 初始化评价指标对象列表
    metric = []
    for metric_name in opt["metric_names"]:
        if metric_name == "DSC":
            metric.append(DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"]))

        elif metric_name == "ASSD":
            metric.append(AverageSymmetricSurfaceDistance_lib(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

        elif metric_name == "HD":
            metric.append(HausdorffDistance_lib(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

        elif metric_name == "SO":
            metric.append(SurfaceOverlappingValues(num_classes=opt["classes"], c=6, theta=1.0))

        else:
            raise Exception(f"{metric_name}是不支持的评价指标！")

    return metric




