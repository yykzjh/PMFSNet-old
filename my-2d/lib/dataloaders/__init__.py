import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from .MMOTUDataset import MMOTUDataset


def get_dataloader(opt):
    """
    获取数据加载器
    Args:
        opt: 参数字典
    Returns:
    """
    if opt["dataset_name"] == "MMOTU":
        # 初始化数据集
        train_set = MMOTUDataset(opt, mode="train")
        valid_set = MMOTUDataset(opt, mode="valid")

        # 初始化数据加载器
        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    elif opt["dataset_name"] == "ILSVRC2012":
        train_set = datasets.ImageFolder(
            os.path.join(opt["dataset_path"], "train"),
            transforms.Compose([
                transforms.RandomResizedCrop(opt["resize_shape"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)

        valid_loader = DataLoader(
            datasets.ImageFolder(os.path.join(opt["dataset_path"], "val"), transforms.Compose([
                transforms.Resize(int(opt["resize_shape"] / 0.875)),
                transforms.CenterCrop(opt["resize_shape"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])),
            batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    # 存储steps_per_epoch
    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader


def get_test_dataloader(opt):
    """
    获取测试集数据加载器
    :param opt: 参数字典
    :return:
    """
    if opt["dataset_name"] == "NCTooth":
        # 初始化数据集
        valid_set = MMOTUDataset(opt, mode="valid")

        # 初始化数据加载器
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    return valid_loader

