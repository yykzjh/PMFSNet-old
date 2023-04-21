# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/19 15:17
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from torch.utils.data import DataLoader

from .NCToothDataset import NCToothDataset


def get_dataloader(opt):
    """
    获取数据加载器
    Args:
        opt: 参数字典
    Returns:
    """
    if opt["dataset_name"] == "NCTooth":
        # 初始化数据集
        train_set = NCToothDataset(opt, mode="train")
        valid_set = NCToothDataset(opt, mode="valid")

        # 初始化数据加载器
        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    return train_loader, valid_loader






