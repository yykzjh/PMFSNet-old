import os
import torch
import argparse
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D


from lib import utils
from lib import dataloaders
from lib import models
from lib import testers
from lib import metrics
import lib.transforms as transforms




# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": -1412,  # clip的下边界数值
    "clip_upper_bound": 17943,  # clip的上边界数值

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    # 标准化均值
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "NCTooth",  # 数据集名称， 可选["NCTooth", ]

    "dataset_path": r"./datasets/NC-release-data-checked",  # 数据集路径

    "batch_size": 1,  # batch_size大小

    "num_workers": 2,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMRFNet",  # 模型名称，可选["DenseVNet","UNet3D", "VNet", "AttentionUNet3D", "R2UNet", "R2AttentionUNet",
    # "HighResNet3D", "DenseVoxelNet", "MultiResUNet3D", "DenseASPPUNet", "PMRFNet", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
    {
        0: "background",
        1: "foreground"
    },

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "metric_names": ["HD", "ASSD", "IoU", "SO", "SD", "DSC"],  # 采用除了dsc之外的评价指标，可选["DSC", "ASSD", "HD", "SO", "SD", "IoU"]

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "dice_mode": "standard",  # DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [32, 32, 32],  # 验证或者测试时滑动分割的移动步长

    "test_type": 2,  # 测试类型，0：单张图像无标签；1：单张图像有标签；2：测试集批量测试

    "single_image_path": None,  # 单张图像的路径

    "single_label_path": None,  # 单张标注图像的路径

}



if __name__ == '__main__':

    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # 获取GPU设备
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("完成初始化配置")

    # 初始化数据变换
    test_transforms = transforms.ComposeTransforms([
        transforms.ClipAndShift(params["clip_lower_bound"], params["clip_upper_bound"]),
        transforms.ToTensor(params["clip_lower_bound"], params["clip_upper_bound"]),
        transforms.Normalize(params["normalize_mean"], params["normalize_std"])
    ])

    # 初始化模型
    model = models.get_model(params)
    print("完成初始化模型")

    # 初始化各评价指标
    metric = metrics.get_metric(params)
    print("完成初始化评价指标")

    print("---------------------------------------------初始化测试器-------------------------------------------------")
    tester = testers.Tester(params, model, metric)


    if params["test_type"] == 0:
        image_np = utils.load_image_or_label(params["single_image_path"], params["resample_spacing"], type="image")
        # 数据变换和数据增强
        image, _ = test_transforms(image_np, None)
        tester.test_single_image_without_label(image)

    elif params["test_type"] == 1:
        image_np = utils.load_image_or_label(params["single_image_path"], params["resample_spacing"], type="image")
        label_np = utils.load_image_or_label(params["single_label_path"], params["resample_spacing"], type="label")
        # 数据变换和数据增强
        image, label = test_transforms(image_np, label_np)
        tester.test_single_image(image, label)

    elif params["test_type"] == 2:
        # 初始化测试集加载器
        test_loader = dataloaders.get_test_dataloader(params)
        print("完成初始化数据加载器")
        tester.test_image_set(test_loader)

















