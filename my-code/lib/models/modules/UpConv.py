# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/2 21:01
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn

from lib.models.modules.ConvBlock import DepthWiseSeparateConvBlock



class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, is_out=False):
        super(UpConv, self).__init__()
        if is_out:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear"),
                nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear"),
                DepthWiseSeparateConvBlock(in_channel=ch_in, out_channel=ch_out, stride=1)
            )

    def forward(self,x):
        x = self.up(x)
        return x






