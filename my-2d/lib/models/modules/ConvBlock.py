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



class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        batch_norm=True,
        preactivation=False,
    ):
        super().__init__()

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = torch.nn.ConstantPad2d(
                tuple([padding % 2, padding - padding % 2] * 2), 0
            )
        else:
            pad = torch.nn.ConstantPad2d(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [torch.nn.BatchNorm2d(in_channel)] + layers
        else:
            layers = [
                pad,
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SingleConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(SingleConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        batch_norm=True,
        preactivation=False,
    ):
        super(DepthWiseSeparateConvBlock, self).__init__()

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = torch.nn.ConstantPad2d(
                tuple([padding % 2, padding - padding % 2] * 2), 0
            )
        else:
            pad = torch.nn.ConstantPad2d(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=True
                )
            ]
            if batch_norm:
                layers = [torch.nn.BatchNorm2d(in_channel)] + layers
        else:
            layers = [
                pad,
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                torch.nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm2d(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)




