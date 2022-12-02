# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/2 20:10
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn


class R2U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=35, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.RRCNN1 = RecurrentResidualBlock(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RecurrentResidualBlock(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RecurrentResidualBlock(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RecurrentResidualBlock(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RecurrentResidualBlock(ch_in=512, ch_out=1024, t=t)

        self.Up4 = UpConv(ch_in=1024, ch_out=512)
        self.Up_RRCNN4 = RecurrentResidualBlock(ch_in=1024, ch_out=512, t=t)

        self.Up3 = UpConv(ch_in=512, ch_out=256)
        self.Up_RRCNN3 = RecurrentResidualBlock(ch_in=512, ch_out=256, t=t)

        self.Up2 = UpConv(ch_in=256, ch_out=128)
        self.Up_RRCNN2 = RecurrentResidualBlock(ch_in=256, ch_out=128, t=t)

        self.Up1 = UpConv(ch_in=128, ch_out=64)
        self.Up_RRCNN1 = RecurrentResidualBlock(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d4 = self.Up4(x5)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_RRCNN1(d1)

        out = self.Conv_1x1(d1)

        return out


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RecurrentResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RecurrentResidualBlock, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class UpConv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpConv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 1, 64, 64, 64)).to(device)

    model = R2U_Net(img_ch=1, output_ch=35, t=2).to(device)

    output = model(x)

    print(x.size())
    print(output.size())

