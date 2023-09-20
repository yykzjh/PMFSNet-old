# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/2 21:05
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.conv(x)
        return x


class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DepthWiseSeparateConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
            nn.BatchNorm3d(in_channel),
            nn.ReLU6(inplace=True),

            nn.Conv3d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)















