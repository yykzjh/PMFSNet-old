from .DiceLoss import *
import torch.nn as nn



def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        loss_function = DiceLoss(opt["classes"], weight=torch.FloatTensor(opt["class_weight"]).to(opt["device"]),
                                 sigmoid_normalization=False, mode=opt["dice_loss_mode"])

    elif opt["loss_function_name"] == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()

    else:
        raise RuntimeError(f"{opt['loss_function_name']}是不支持的损失函数！")

    return loss_function