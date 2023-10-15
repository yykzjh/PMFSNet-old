# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/3 20:02
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import torch
import torch.nn as nn

from collections import OrderedDict

from lib.models.modules.ConvBlock import DepthWiseSeparateConvBlock, SingleConvBlock, ConvBlock



class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector(nn.Module):
    """
    使用多尺度感受野信息扩充内积向量长度的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, kernels=[1, 3, 5], group=1, r=4):
        """
        定义一个使用多尺度感受野信息扩充内积向量长度的局部极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param r: 衰减率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector, self).__init__()
        self.inner_c = max(4, channel // (len(kernels) * r))
        self.ks = [3 if k > 1 else 1 for k in kernels]
        self.pads = [(k - 1) // 2 for k in kernels]
        self.dils = [1 if k < 3 else ((k - 1) // 2) for k in kernels]
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 定义Wq的不同感受野分支的卷积
        self.ch_Wq_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i], groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义Wk的不同感受野分支的卷积
        self.ch_Wk_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, 1, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i])),
                ("bn", nn.BatchNorm3d(1)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.inner_c, channel, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((channel, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义Wq的不同感受野分支的卷积
        self.sp_Wq_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i], groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义Wk的不同感受野分支的卷积
        self.sp_Wk_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i], groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)



    def forward(self, x):
        # 先变换通道数
        x = self.basic_conv(x)
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 计算通道Wq各分支
        ch_Wq_outs = [
            conv(x)
            for conv in self.ch_Wq_convs
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_Wq_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 计算通道Wk各分支
        ch_Wk_outs = [
            conv(x)
            for conv in self.ch_Wk_convs
        ]
        # 堆叠通道Wk
        ch_Wk = torch.stack(ch_Wk_outs, dim=-1)  # bs, 1, d, h, w, k
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, self.inner_c, -1)  # bs, self.inner_c, d*h*w*k
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w*k, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w*k, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz)))  # bs, c, 1, 1, 1
        # 通道增强
        ch_out = ch_score * x

        # 计算空间Wq各分支
        sp_Wq_outs = [
            conv(ch_out)
            for conv in self.sp_Wq_convs
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_Wq_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 计算空间Wk各分支
        sp_Wk_outs = [
            conv(ch_out)
            for conv in self.sp_Wk_convs
        ]
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_Wk_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        sp_Wq = sp_Wq.reshape(bs, -1, d * h * w)  # bs, k*self.inner_c, d*h*w
        sp_Wk = sp_Wk.mean(-1).mean(-1).mean(-1).reshape(bs, 1, -1)  # bs, 1, k*self.inner_c
        # 进行Softmax处理
        sp_Wk = self.sp_softmax(sp_Wk)  # bs, 1, k*self.inner_c
        # 矩阵相乘
        sp_Wz = torch.matmul(sp_Wk, sp_Wq)  # bs, 1, d*h*w
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w))  # bs, 1, d, h, w
        # 空间增强
        sp_out = sp_score * ch_out

        return sp_out


class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints(nn.Module):
    """
    使用多尺度感受野信息扩充注意力位置的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, kernels=[1, 3, 5], group=1, r=4):
        """
        定义一个使用多尺度感受野信息扩充注意力位置的局部极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param r: 衰减率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints, self).__init__()
        self.inner_c = max(4, channel // (len(kernels) * r))
        self.k = len(kernels)
        self.ks = [3 if k > 1 else 1 for k in kernels]
        self.pads = [(k - 1) // 2 for k in kernels]
        self.dils = [1 if k < 3 else ((k-1) // 2) for k in kernels]
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i],
                                   groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义通道Wv的不同感受野分支的卷积
        self.ch_Wv_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv",
                 nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i],
                           groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.k * self.inner_c, self.k * self.inner_c, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((self.k * self.inner_c, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # 通道注意力恢复通道数
        self.ch_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)

        # 定义空间的不同感受野分支的卷积
        self.sp_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i],
                                   groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义空间Wv的不同感受野分支的卷积
        self.sp_Wv_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv",
                 nn.Conv3d(channel, self.inner_c, kernel_size=self.ks[i], padding=self.pads[i], dilation=self.dils[i],
                           groups=group, bias=False)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for i, k in enumerate(kernels)
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)
        # 空间注意力恢复通道数
        self.sp_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)



    def forward(self, x):
        # 先变换通道数
        x = self.basic_conv(x)
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()


        # 计算通道各分支
        ch_outs = [
            conv(x)
            for conv in self.ch_convs
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 堆叠通道Wk
        ch_Wk = torch.cat(ch_outs, dim=1).mean(dim=1, keepdim=True)  # bs, 1, d, h, w
        # 计算通道Wv各分支
        ch_Wv_outs = [
            conv(x)
            for conv in self.ch_Wv_convs
        ]
        # 堆叠通道Wv
        ch_Wv = torch.stack(ch_Wv_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, -1, d * h * w)  # bs, k * self.inner_c, d*h*w
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, k * self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz))).reshape(bs, -1, self.inner_c, 1, 1, 1)  # bs, k, self.inner_c, 1, 1, 1
        # 通道增强
        ch_out = torch.sum(ch_score * ch_Wv, dim=1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        ch_out = self.ch_excit(ch_out)  # bs, c, d, h, w

        # 计算空间各分支
        sp_outs = [
            conv(ch_out)
            for conv in self.sp_convs
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 计算空间Wv各分支
        sp_Wv_outs = [
            conv(x)
            for conv in self.sp_Wv_convs
        ]
        # 堆叠通道Wv
        sp_Wv = torch.stack(sp_Wv_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 转换维度
        sp_Wq = sp_Wq.reshape(bs, self.inner_c, -1)  # bs, self.inner_c, d*h*w*k
        sp_Wk = sp_Wk.mean(-1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, -1)  # bs, 1, self.inner_c
        # 进行Softmax处理
        sp_Wk = self.sp_softmax(sp_Wk)  # bs, 1, self.inner_c
        # 矩阵相乘
        sp_Wz = torch.matmul(sp_Wk, sp_Wq)  # bs, 1, d*h*w*k
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w, self.k))  # bs, 1, d, h, w, k
        # 空间增强
        sp_out = torch.sum(sp_score * sp_Wv, dim=-1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        sp_out = self.sp_excit(sp_out)  # bs, c, d, h, w

        return sp_out



class LocalPMFSBlock_AP(nn.Module):
    """
    使用多尺度特征扩充注意力关注点数量，从而对各尺度特征进行增强的极化多尺度特征自注意力模块
    """
    def __init__(self, ch, ch_k, ch_v, br):
        """
        定义一个极化多尺度特征自注意力模块

        :param ch: 输入通道数，也是输出的通道数
        :param ch_k: K的通道数
        :param ch_v: V的通道数
        :param br: 多尺度特征的数量
        """
        super(LocalPMFSBlock_AP, self).__init__()
        # 初始化参数
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_in = self.ch * self.br

        # 定义通道Wq卷积
        self.ch_Wq = SingleConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1)
        # 定义通道Wk卷积
        self.ch_Wk = SingleConvBlock(in_channel=self.ch_in, out_channel=1, kernel_size=1, stride=1)
        # 定义通道Wv卷积
        self.ch_Wv = SingleConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1)
        # 定义通道K的softmax
        self.ch_softmax = nn.Softmax(dim=1)
        # 定义对通道分数矩阵的卷积
        self.ch_score_conv = nn.Conv3d(self.ch_in, self.ch_in, kernel_size=1)
        # 定义对通道分数矩阵的LayerNorm层归一化
        self.ch_layer_norm = nn.LayerNorm((self.ch_in, 1, 1, 1))
        # 定义sigmoid
        self.sigmoid = nn.Sigmoid()
        # # 定义通道输出前的前馈卷积
        # self.ch_forward_conv = nn.Conv3d(self.ch_in, self.ch_in, kernel_size=5, stride=1, padding=2, groups=self.ch_in, bias=True)

        # 定义空间Wq卷积
        self.sp_Wq = SingleConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1)
        # 定义空间Wk卷积
        self.sp_Wk = SingleConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1)
        # 定义空间Wv卷积
        self.sp_Wv = SingleConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_v, kernel_size=1, stride=1)
        # 定义空间K的softmax
        self.sp_softmax = nn.Softmax(dim=-1)
        # 定义空间卷积，还原通道数
        self.sp_output_conv = SingleConvBlock(in_channel=self.br * self.ch_v, out_channel=self.ch_in, kernel_size=3, stride=1)
        # # 定义空间输出前的前馈卷积
        # self.sp_forward_conv = nn.Conv3d(self.ch_in, self.ch_in, kernel_size=1, stride=1, groups=self.br)

    def forward(self, x):
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 先计算通道ch_Q、ch_K、ch_V
        ch_Q = self.ch_Wq(x)  # bs, self.ch_in, d, h, w
        ch_K = self.ch_Wk(x)  # bs, 1, d, h, w
        ch_V = self.ch_Wv(x)  # bs, self.ch_in, d, h, w
        # 转换通道ch_Q维度
        ch_Q = ch_Q.reshape(bs, -1, d * h * w)  # bs, self.ch_in, d*h*w
        # 转换通道ch_K维度
        ch_K = ch_K.reshape(bs, -1, 1)  # bs, d*h*w, 1
        # 对通道ch_K采取softmax
        ch_K = self.ch_softmax(ch_K)  # bs, d*h*w, 1
        # 将通道ch_Q和通道ch_K相乘
        Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1).unsqueeze(-1)  # bs, self.ch_in, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1, 1
        # 通道增强
        ch_out = ch_V * ch_score  # bs, self.ch_in, d, h, w
        # # 拷贝一份通道的残差特征
        # ch_residual = ch_out  # bs, self.ch_in, d, h, w
        # # 通道输出前的前馈卷积
        # ch_out = self.ch_forward_conv(ch_out)  # bs, self.ch_in, d, h, w
        # # 通道残差相加
        # ch_out = ch_out + ch_residual  # bs, self.ch_in, d, h, w

        # 先计算空间sp_Q、sp_K、sp_V
        sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, d, h, w
        sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, d, h, w
        sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, d, h, w
        # 转换空间sp_Q维度
        sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, d*h*w*self.br
        # 转换空间sp_K维度
        sp_K = sp_K.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1).mean(-1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
        # 转换空间sp_V维度
        sp_V = sp_V.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1)  # bs, self.ch_v, d, h, w, self.br
        # 对空间sp_K采取softmax
        sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
        # 将空间sp_K和空间sp_Q相乘
        Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, d, h, w, self.br)  # bs, 1, d, h, w, self.br
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(Z)  # bs, 1, d, h, w, self.br
        # 空间增强
        sp_out = sp_V * sp_score  # bs, self.ch_v, d, h, w, self.br
        # 变换空间增强后的维度
        sp_out = sp_out.permute(0, 5, 1, 2, 3, 4).reshape(bs, self.br * self.ch_v, d, h, w)  # bs, self.br*self.ch_v, d, h, w
        # 还原通道数
        sp_out = self.sp_output_conv(sp_out)
        # # 拷贝一份空间的残差特征
        # sp_residual = sp_out  # bs, self.ch_in, d, h, w
        # # 最后输出前的卷积，还原通道数
        # sp_out = self.sp_forward_conv(sp_out)  # bs, self.ch_in, d, h, w
        # # 通道残差相加
        # sp_out = sp_out + sp_residual  # bs, self.ch_in, d, h, w

        return sp_out



# class DenseConvWithPMFSBlock(nn.Module):
#     """
#     带有极化多尺度特征增强自注意力模块的密集卷积块
#     """
#     def __init__(self, in_ch, out_ch, dilations=(1, 2, 3), r=4):
#         """
#         定义一个带有极化多尺度特征增强自注意力模块的密集卷积块
#
#         :param in_ch: 输入通道数
#         :param out_ch: 输出通道数
#         :param dilations: 各卷积层空洞率，长度表明堆叠次数
#         :param r: 内部通道数相对于输出通道数的衰减率
#         """
#         super(DenseConvWithPMFSBlock, self).__init__()
#         # 初始化参数
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.dilations = dilations
#         self.inner_ch = self.out_ch // r
#         self.layer_num = len(self.dilations)
#         # 将输入特征通道压缩
#         self.input_conv = nn.Conv3d(self.in_ch, self.inner_ch, kernel_size=1, stride=1)
#         # 定义多次的堆叠密集卷积层，每层包括空洞卷积、BN、ReLU、PMFSBlock
#         self.dense_conv_layers = nn.ModuleList([
#             nn.Sequential(OrderedDict([
#                 ("conv", nn.Conv3d((i+1)*self.inner_ch, self.inner_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False)),
#                 ("bn", nn.BatchNorm3d(self.inner_ch)),
#                 ("relu", nn.ReLU(inplace=True)),
#                 ("pmfs", PMFSBlock_AP(self.inner_ch, self.inner_ch//2, self.inner_ch//2, i+2))
#             ]))
#             for i, dilation in enumerate(self.dilations)
#         ])
#         # 输出前将通道数卷积到输出值
#         self.output_conv = nn.Sequential(
#             nn.Conv3d((self.layer_num + 1) * self.inner_ch, self.out_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm3d(self.out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#
#     def forward(self, x):
#         # 获得输入特征图维度信息
#         bs, c, d, h, w = x.size()
#
#         # 将输入特征通道压缩
#         x = self.input_conv(x)
#
#         # 遍历计算堆叠密集卷积层
#         for i, layer in enumerate(self.dense_conv_layers):
#             # 空洞卷积+bn+relu
#             y = layer.conv(x)
#             y = layer.bn(y)
#             y = layer.relu(y)
#             # concat
#             x = torch.cat([x, y], dim=1)
#             # pmfs特征增强
#             x = layer.pmfs(x)
#
#         # 输出前的卷积层
#         out = self.output_conv(x)
#
#         return out


class DenseFeatureStackWithLocalPMFSBlock(nn.Module):

    def __init__(self, in_channel, kernel_size, unit, growth_rate):
        """
        定义一个带有局部极化多尺度特征增强自注意力模块的密集卷积块

        :param in_channel: 输入通道数
        :param kernel_size: 卷积核大小
        :param unit: 密集堆叠单元个数
        :param growth_rate: 每次堆叠增加的通道数
        """
        super(DenseFeatureStackWithLocalPMFSBlock, self).__init__()

        self.conv_units = torch.nn.ModuleList()
        # self.pmfs_units = torch.nn.ModuleList()
        for i in range(unit):
            self.conv_units.append(
                ConvBlock(
                    in_channel=in_channel,
                    out_channel=growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    batch_norm=True,
                    preactivation=True
                )
            )
            # self.pmfs_units.append(
            #     LocalPMFSBlock_AP(
            #         ch=growth_rate,
            #         ch_k=growth_rate,
            #         ch_v=growth_rate,
            #         br=i+1
            #     )
            # )
            in_channel += growth_rate

    def forward(self, x):
        stack_feature = None

        for i, conv in enumerate(self.conv_units):
            if stack_feature is None:
                inputs = x
            else:
                inputs = torch.cat([x, stack_feature], dim=1)
            out = conv(inputs)
            if stack_feature is None:
                stack_feature = out
            else:
                stack_feature = torch.cat([stack_feature, out], dim=1)
            # stack_feature = self.pmfs_units[i](stack_feature)

        return torch.cat([x, stack_feature], dim=1)


class DownSampleWithLocalPMFSBlock(nn.Module):
    """
    带有局部极化多尺度特征增强自注意力模块的下采样模块
    """
    def __init__(self, in_channel, base_channel, kernel_size, unit, growth_rate, skip_channel=None, downsample=True, skip=True):
        """
        定义一个带有局部极化多尺度特征增强自注意力模块的下采样模块

        :param in_channel: 输入通道数
        :param base_channel: 基础通道数
        :param kernel_size: 卷积核大小
        :param unit: 密集堆叠单元个数
        :param growth_rate: 每次堆叠增加的通道数
        :param skip_channel: 跳跃连接通道数
        :param downsample: 是否下采样
        :param skip: 是否产生跳跃连接特征图
        """
        super(DownSampleWithLocalPMFSBlock, self).__init__()
        self.skip = skip

        self.downsample = ConvBlock(
            in_channel=in_channel,
            out_channel=base_channel,
            kernel_size=kernel_size,
            stride=(2 if downsample else 1),
            batch_norm=True,
            preactivation=True
        )

        self.dfs_with_pmfs = DenseFeatureStackWithLocalPMFSBlock(
            in_channel=base_channel,
            kernel_size=3,
            unit=unit,
            growth_rate=growth_rate
        )

        if skip:
            self.skip_conv = ConvBlock(
                in_channel=base_channel + unit * growth_rate,
                out_channel=skip_channel,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True
            )

    def forward(self, x):
        x = self.downsample(x)
        x = self.dfs_with_pmfs(x)

        if self.skip:
            x_skip = self.skip_conv(x)
            return x, x_skip
        else:
            return x




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 64, 32, 32, 32)).to(device)

    model = DownSampleWithLocalPMFSBlock(64, 128, 10, 16).to(device)

    output = model(x)

    print(x.size())
    print(output.size())
















