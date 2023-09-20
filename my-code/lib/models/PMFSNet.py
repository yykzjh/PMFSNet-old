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
from lib.models.modules.LocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP



class PMFSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=35,
                 basic_module=DownSampleWithLocalPMFSBlock,
                 global_module=GlobalPMFSBlock_AP):
        super(PMFSNet, self).__init__()

        downsample_channels = [32, 64, 128]
        units = [5, 10, 10]
        growth_rates = [4, 8, 16]

        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                basic_module(
                    in_channel=(1 if i == 0 else downsample_channels[i-1]),
                    out_channel=downsample_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i]
                )
            )

        self.Global = global_module(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=64,
            ch_k=64,
            ch_v=64,
            br=3
        )

        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.up_conv2 = basic_module(in_channel=128, out_channel=64, unit=units[1], growth_rate=growth_rates[1], downsample=False)

        self.up1 = UpConv(ch_in=64, ch_out=32)
        self.up_conv1 = basic_module(in_channel=64, out_channel=32, unit=units[0], growth_rate=growth_rates[0], downsample=False)

        self.out_conv = UpConv(ch_in=32, ch_out=out_channels, is_out=True)



    def forward(self, x):
        # encoding
        x1 = self.down_convs[0](x)
        x2 = self.down_convs[1](x1)
        x3 = self.down_convs[2](x2)

        d3 = self.Global([x1, x2, x3])

        # decoding + concat
        d2 = self.up2(d3)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.up_conv1(d1)

        out = self.out_conv(d1)

        return out





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 1, 160, 160, 96)).to(device)

    model = PMFSNet(in_channels=1, out_channels=2).to(device)

    output = model(x)

    print(x.size())
    print(x.device)
    print(output.size())








