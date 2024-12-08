# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 14:53
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from .DICE import *
from .IoU import *
from .JI import *
from .ACC import *


def get_metric(opt):
    # 初始化评价指标对象列表
    metric = {}
    for metric_name in opt["metric_names"]:
        if metric_name == "DSC":
            metric[metric_name] = DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"])

        elif metric_name == "IoU":
            metric[metric_name] = IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

        elif metric_name == "JI":
            metric[metric_name] = JI(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

        elif metric_name == "ACC":
            metric[metric_name] = ACC(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

        else:
            raise Exception(f"{metric_name}是不支持的评价指标！")

    return metric
