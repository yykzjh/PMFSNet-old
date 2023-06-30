import os
import torch
import argparse
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import configs.config as config

from lib import utils
from lib import dataloaders
from lib import models
from lib import testers
from lib import metrics




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default=None, type=str, metavar='PATH',
                        help='path to latest pretrain model (default: None)')
    args = parser.parse_args()

    return args




def main():
    print("-----------------------------------------------解析参数--------------------------------------------------")
    args = get_arguments()

    print("------------------------------------设置可用GPU、随机种子、卷积算法优化---------------------------------------")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    utils.reproducibility()

    print("--------------------------------------------初始化预训练模型-----------------------------------------------")
    model = models.create_test_model(args.pretrain)

    if config.cuda:
        print("--------------------------------------------将模型放到GPU上-----------------------------------------------")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    print("---------------------------------------------初始化评价指标------------------------------------------------")
    meter = metrics.BaseMeter(classes=config.classes, softmax_normalization=True)

    print("---------------------------------------------初始化测试器-------------------------------------------------")
    tester = testers.Tester(model, meter, device=device)

    if config.test_type == 0:
        image = dataloaders.load_image_or_label(config.single_image_path, type="image", viz3d=False)
        tester.test_single_image_without_label(image)

    elif config.test_type == 1:
        image = dataloaders.load_image_or_label(config.single_image_path, type="image", viz3d=False)
        label = dataloaders.load_image_or_label(config.single_label_path, type="label", viz3d=False)
        tester.test_single_image(image, label)

    elif config.test_type == 2:
        print("--------------------------------------------初始化数据加载器-----------------------------------------------")
        test_loader = dataloaders.get_test_loader()
        tester.test_image_set(test_loader)







if __name__ == '__main__':
    main()
















