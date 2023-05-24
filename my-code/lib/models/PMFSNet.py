# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/5 23:25
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from lib.models.modules.UpConv import UpConv
from lib.models.modules.RecurrentResidualBlock import RecurrentResidualBlock
from lib.models.modules.GridAttentionGate3d import GridAttentionGate3d
from lib.models.modules.LocalPMFSBlock import DenseConvWithLocalPMFSBlock
from lib.models.modules.GlobalPMFSBlock import GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector, \
    GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints



class PMFSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=35,
                 basic_module=DenseConvWithLocalPMFSBlock,
                 global_module=GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints):
        super(PMFSNet, self).__init__()

        self.Local1 = basic_module(in_channels, 32)

        self.down1 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1)
        self.Local2 = basic_module(32, 64)

        self.down2 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.Local3 = basic_module(64, 128)

        self.down3 = nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1)

        self.Global = global_module(128, 256, feature_num=4)

        self.Up3 = UpConv(ch_in=256, ch_out=128)
        self.Att3 = GridAttentionGate3d(F_l=128, F_g=256, F_int=64)
        self.Up_Local3 = basic_module(256, 128)

        self.Up2 = UpConv(ch_in=128, ch_out=64)
        self.Att2 = GridAttentionGate3d(F_l=64, F_g=128, F_int=32)
        self.Up_Local2 = basic_module(128, 64)

        self.Up1 = UpConv(ch_in=64, ch_out=32)
        self.Att1 = GridAttentionGate3d(F_l=32, F_g=64, F_int=16)
        self.Up_Local1 = basic_module(64, 32)

        self.Conv_1x1 = nn.Conv3d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Local1(x)

        x2 = self.down1(x1)
        x2 = self.Local2(x2)

        x3 = self.down2(x2)
        x3 = self.Local3(x3)

        x4 = self.down3(x3)

        d4 = self.Global([x4, x3, x2, x1])

        # decoding + concat path
        d3 = self.Up3(d4)
        x3 = self.Att3(x=x3, g=d4)[0]
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_Local3(d3)

        d2 = self.Up2(d3)
        x2 = self.Att2(x=x2, g=d3)[0]
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_Local2(d2)

        d1 = self.Up1(d2)
        x1 = self.Att1(x=x1, g=d2)[0]
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.Up_Local1(d1)

        out = self.Conv_1x1(d1)

        return out





if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 1, 160, 160, 96)).to(device)

    model = PMFSNet(in_channels=1, out_channels=2).to(device)

    output = model(x)

    print(x.size())
    print(x.device)
    print(output.size())








