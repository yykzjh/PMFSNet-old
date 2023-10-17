from .DiceLoss import *



def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        loss_function = DiceLoss(opt["classes"], weight=torch.FloatTensor(opt["class_weight"]).to(opt["device"]),
                                 sigmoid_normalization=False, mode=opt["dice_loss_mode"])

    else:
        raise RuntimeError(f"{opt['loss_function_name']}是不支持的损失函数！")

    return loss_function