# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/1 18:26
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch

def expand_as_one_hot(input, C):
    """
    Converts NxHxW label image to NxCxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 3D input image (NxHxW)
    :param C: number of channels/labels
    :return: 4D output image (NxCxHxW)
    """
    if input.dim() == 4:
        return input
    assert input.dim() == 3

    # expand the input tensor to Nx1xHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxHxW)
    shape = list(input.size())
    shape[1] = C

    # scatter to get the one-hot tensor
    return torch.zeros(shape).to(input.device).scatter_(1, input, 1)



