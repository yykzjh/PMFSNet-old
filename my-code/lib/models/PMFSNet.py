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
from lib.models.modules.ConvBlock import ConvBlock, DepthWiseSeparateConvBlock
from lib.models.modules.RecurrentResidualBlock import RecurrentResidualBlock
from lib.models.modules.GridAttentionGate3d import GridAttentionGate3d
from lib.models.modules.LocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP, GlobalPMFSBlock_AP_Separate



class PMFSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=35,
                 basic_module=DownSampleWithLocalPMFSBlock,
                 global_module=GlobalPMFSBlock_AP_Separate):
        super(PMFSNet, self).__init__()

        kernel_sizes = [5, 3, 3]
        base_channels = [24, 24, 24]
        skip_channels = [12, 24, 24]
        units = [5, 10, 10]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]  # [44, 104, 184]

        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                basic_module(
                    in_channel=(in_channels if i == 0 else downsample_channels[i-1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    skip=True
                )
            )

        self.Global = global_module(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=48,
            ch_k=48,
            ch_v=48,
            br=3
        )

        self.bottle_conv = ConvBlock(
            in_channel=downsample_channels[2] + skip_channels[2],
            out_channel=skip_channels[2],
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
        )

        # self.up2 = UpConv(ch_in=downsample_channels[2], ch_out=downsample_channels[1])
        # self.up_conv2 = basic_module(in_channel=downsample_channels[1] + skip_channels[1], base_channel=base_channels[1], unit=units[1], growth_rate=growth_rates[1], downsample=False, skip=False)
        #
        # self.up1 = UpConv(ch_in=downsample_channels[1], ch_out=downsample_channels[0])
        # self.up_conv1 = basic_module(in_channel=downsample_channels[0] + skip_channels[0], base_channel=base_channels[0], unit=units[0], growth_rate=growth_rates[0], downsample=False, skip=False)
        #
        # self.out_conv = UpConv(ch_in=downsample_channels[0], ch_out=out_channels, is_out=True)


        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode='trilinear')

        self.out_conv = ConvBlock(
            in_channel=sum(skip_channels),
            out_channel=out_channels,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode='trilinear')



    def forward(self, x):
        # # encoding
        # x1, x1_skip = self.down_convs[0](x)
        # x2, x2_skip = self.down_convs[1](x1)
        # x3 = self.down_convs[2](x2)
        #
        # d3 = self.Global([x1, x2, x3])
        #
        # # decoding + concat
        # d2 = self.up2(d3)
        # d2 = torch.cat((x2_skip, d2), dim=1)
        # d2 = self.up_conv2(d2)
        #
        # d1 = self.up1(d2)
        # d1 = torch.cat((x1_skip, d1), dim=1)
        # d1 = self.up_conv1(d1)
        #
        # out = self.out_conv(d1)
        #
        # return out


        # encoding
        x1, skip1 = self.down_convs[0](x)
        x2, skip2 = self.down_convs[1](x1)
        x3, skip3 = self.down_convs[2](x2)

        x3 = self.Global([x1, x2, x3])
        skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))

        skip2 = self.upsample_1(skip2)
        skip3 = self.upsample_2(skip3)

        out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
        out = self.upsample_out(out)

        return out





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 1, 160, 160, 96)).to(device)

    model = PMFSNet(in_channels=1, out_channels=2).to(device)

    output = model(x)

    print(x.size())
    print(x.device)
    print(output.size())








