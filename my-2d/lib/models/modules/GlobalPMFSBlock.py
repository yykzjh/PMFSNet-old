# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/4 21:07
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import torch
import torch.nn as nn

from collections import OrderedDict

from lib.models.modules.ConvBlock import DepthWiseSeparateConvBlock, SingleConvBlock




class GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector(nn.Module):
    """
    使用多尺度感受野信息扩充内积向量长度的全局极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, feature_num=5, r=4):
        """
        定义一个使用多尺度感受野信息扩充内积向量长度的全局极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param feature_num: 不同层次的特征图数量
        :param r: 衰减率
        """
        super(GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector, self).__init__()
        self.inner_c = channel // (feature_num * r)
        self.k = feature_num
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 先确定每个特征图对应的最大池化层的核以及步长大小
        self.max_pool_size = [2**i for i in range(feature_num)]
        # 定义最大池化层
        self.max_pool_layers = nn.ModuleList([
            nn.MaxPool3d(kernel_size=k, stride=k)
            for k in self.max_pool_size
        ])

        # 定义通道Wq的不同感受野分支的卷积
        self.ch_Wq_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义通道Wk的不同感受野分支的卷积
        self.ch_Wk_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, 1, kernel_size=1)),
                ("bn", nn.BatchNorm3d(1)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.inner_c, channel, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((channel, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义空间Wq的不同感受野分支的卷积
        self.sp_Wq_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义空间Wk的不同感受野分支的卷积
        self.sp_Wk_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)



    def forward(self, feature_maps):
        # 先变换通道数
        feature_maps[0] = self.basic_conv(feature_maps[0])
        # 获得输入特征图维度信息
        bs, c, d, h, w = feature_maps[0].size()
        # 用最大池化统一特征图尺寸
        max_pool_maps = [
            self.max_pool_layers[i](feature_maps[i])
            for i in range(self.k)
        ]

        # 计算通道Wq各分支
        ch_Wq_outs = [
            self.ch_Wq_convs[i](max_pool_maps[i])
            for i in range(self.k)
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_Wq_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 计算通道Wk各分支
        ch_Wk_outs = [
            self.ch_Wk_convs[i](max_pool_maps[i])
            for i in range(self.k)
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
        max_pool_maps[0] = ch_score * max_pool_maps[0]

        # 计算空间Wq各分支
        sp_Wq_outs = [
            self.sp_Wq_convs[i](max_pool_maps[i])
            for i in range(self.k)
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_Wq_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 计算空间Wk各分支
        sp_Wk_outs = [
            self.sp_Wk_convs[i](max_pool_maps[i])
            for i in range(self.k)
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
        max_pool_maps[0] = sp_score * max_pool_maps[0]

        return max_pool_maps[0]



class GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints(nn.Module):
    """
    使用多尺度感受野信息扩充注意力位置的全局极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, feature_num=5, r=4):
        """
        定义一个使用多尺度感受野信息扩充注意力位置的全局极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param feature_num: 不同层次的特征图数量
        :param r: 衰减率
        """
        super(GlobalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints, self).__init__()
        self.inner_c = channel // (feature_num * r)
        self.k = feature_num
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 先确定每个特征图对应的最大池化层的核以及步长大小
        self.max_pool_size = [2 ** i for i in range(feature_num)]
        # 定义最大池化层
        self.max_pool_layers = nn.ModuleList([
            nn.MaxPool3d(kernel_size=k, stride=k)
            for k in self.max_pool_size
        ])

        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义通道Wv的不同感受野分支的卷积
        self.ch_Wv_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
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
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义空间Wv的不同感受野分支的卷积
        self.sp_Wv_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel//k, self.inner_c, kernel_size=1)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in self.max_pool_size
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)
        # 空间注意力恢复通道数
        self.sp_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)



    def forward(self, feature_maps):
        # 先变换通道数
        feature_maps[0] = self.basic_conv(feature_maps[0])
        # 获得输入特征图维度信息
        bs, c, d, h, w = feature_maps[0].size()
        # 用最大池化统一特征图尺寸
        max_pool_maps = [
            self.max_pool_layers[i](feature_maps[i])
            for i in range(self.k)
        ]

        # 计算通道各分支
        ch_outs = [
            self.ch_convs[i](max_pool_maps[i])
            for i in range(self.k)
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 堆叠通道Wk
        ch_Wk = torch.cat(ch_outs, dim=1).mean(dim=1, keepdim=True)  # bs, 1, d, h, w
        # 计算通道Wv各分支
        ch_Wv_outs = [
            self.ch_Wv_convs[i](max_pool_maps[i])
            for i in range(self.k)
        ]
        # 堆叠通道Wv
        ch_Wv = torch.stack(ch_Wv_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, -1, d * h * w)  # bs, k * self.inner_c, d*h*w
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w*k, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, k * self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz))).reshape(bs, -1, self.inner_c, 1, 1, 1)  # bs, k, self.inner_c, 1, 1, 1
        # 通道增强
        ch_out = torch.sum(ch_score * ch_Wv, dim=1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        max_pool_maps[0] = self.ch_excit(ch_out)  # bs, c, d, h, w

        # 计算空间各分支
        sp_outs = [
            self.sp_convs[i](max_pool_maps[i])
            for i in range(self.k)
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 计算空间Wv各分支
        sp_Wv_outs = [
            self.sp_Wv_convs[i](max_pool_maps[i])
            for i in range(self.k)
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



class GlobalPMFSBlock_AP(nn.Module):
    """
    使用全局的多尺度特征扩充注意力关注点数量，从而对各尺度特征进行增强的全局极化多尺度特征自注意力模块
    """
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br):
        """
        定义一个全局的极化多尺度特征自注意力模块

        :param in_channels: 输入各尺度特征图的通道数
        :param max_pool_kernels: 输入各尺度特征图的下采样核大小
        :param ch: 全局特征统一的通道数
        :param ch_k: K的通道数
        :param ch_v: V的通道数
        :param br: 多尺度特征的数量
        """
        super(GlobalPMFSBlock_AP, self).__init__()
        # 初始化参数
        self.ch_bottle = in_channels[-1]
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_in = self.ch * self.br

        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            SingleConvBlock(
                in_channel=in_channel,
                out_channel=self.ch,
                kernel_size=3,
                stride=1
            )
            for in_channel in in_channels
        ])

        # 定义将全局特征下采样到相同尺寸的最大池化层
        self.max_pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=k)
            for k in max_pool_kernels
        ])

        # 定义通道Wq卷积
        self.ch_Wq = SingleConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1)
        # 定义通道Wk卷积
        self.ch_Wk = SingleConvBlock(in_channel=self.ch_in, out_channel=1, kernel_size=1, stride=1)
        # 定义通道Wv卷积
        self.ch_Wv = SingleConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1)
        # 定义通道K的softmax
        self.ch_softmax = nn.Softmax(dim=1)
        # 定义对通道分数矩阵的卷积
        self.ch_score_conv = nn.Conv2d(self.ch_in, self.ch_in, 1)
        # 定义对通道分数矩阵的LayerNorm层归一化
        self.ch_layer_norm = nn.LayerNorm((self.ch_in, 1, 1))
        # 定义sigmoid
        self.sigmoid = nn.Sigmoid()

        # 定义空间Wq卷积
        self.sp_Wq = SingleConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1)
        # 定义空间Wk卷积
        self.sp_Wk = SingleConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1)
        # 定义空间Wv卷积
        self.sp_Wv = SingleConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_v, kernel_size=1, stride=1)
        # 定义空间K的softmax
        self.sp_softmax = nn.Softmax(dim=-1)
        # 定义空间卷积，还原通道数
        self.sp_output_conv = SingleConvBlock(in_channel=self.br * self.ch_v, out_channel=self.ch_in, kernel_size=1, stride=1)

        # 定义输出卷积，将通道数转换为输入的瓶颈层特征图通道数
        self.output_conv = SingleConvBlock(in_channel=self.ch_in, out_channel=self.ch_bottle, kernel_size=3, stride=1)

    def forward(self, feature_maps):
        # 用最大池化统一特征图尺寸
        max_pool_maps = [
            max_pool_layer(feature_maps[i])
            for i, max_pool_layer in enumerate(self.max_pool_layers)
        ]
        # 计算通道各分支
        ch_outs = [
            ch_conv(max_pool_maps[i])
            for i, ch_conv in enumerate(self.ch_convs)
        ]
        # 将不同分支的特征进行拼接
        x = torch.cat(ch_outs, dim=1)  # bs, self.ch_in, h, w
        # 获得拼接后特征图的维度信息
        bs, c, h, w = x.size()

        # 先计算通道ch_Q、ch_K、ch_V
        ch_Q = self.ch_Wq(x)  # bs, self.ch_in, h, w
        ch_K = self.ch_Wk(x)  # bs, 1, d, h, w
        ch_V = self.ch_Wv(x)  # bs, self.ch_in, h, w
        # 转换通道ch_Q维度
        ch_Q = ch_Q.reshape(bs, -1, h * w)  # bs, self.ch_in, h*w
        # 转换通道ch_K维度
        ch_K = ch_K.reshape(bs, -1, 1)  # bs, h*w, 1
        # 对通道ch_K采取softmax
        ch_K = self.ch_softmax(ch_K)  # bs, h*w, 1
        # 将通道ch_Q和通道ch_K相乘
        Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1)  # bs, self.ch_in, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1
        # 通道增强
        ch_out = ch_V * ch_score  # bs, self.ch_in, h, w

        # 先计算空间sp_Q、sp_K、sp_V
        sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, h, w
        sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, h, w
        sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, h, w
        # 转换空间sp_Q维度
        sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, h*w*self.br
        # 转换空间sp_K维度
        sp_K = sp_K.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
        # 转换空间sp_V维度
        sp_V = sp_V.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)  # bs, self.ch_v, d, h, w, self.br
        # 对空间sp_K采取softmax
        sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
        # 将空间sp_K和空间sp_Q相乘
        Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, h, w, self.br)  # bs, 1, h, w, self.br
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(Z)  # bs, 1, h, w, self.br
        # 空间增强
        sp_out = sp_V * sp_score  # bs, self.ch_v, h, w, self.br
        # 变换空间增强后的维度
        sp_out = sp_out.permute(0, 4, 1, 2, 3).reshape(bs, self.br * self.ch_v, h, w)  # bs, self.br*self.ch_v, d, h, w
        # 还原通道数
        sp_out = self.sp_output_conv(sp_out)  # bs, self.ch_in, h, w

        # 最终的输出卷积，将通道数转换为输入的瓶颈层特征图通道数
        out = self.output_conv(sp_out)

        return out


class GlobalPMFSBlock_AP_Separate(nn.Module):
    """
    使用全局的多尺度特征扩充注意力关注点数量，从而对各尺度特征进行增强的全局极化多尺度特征自注意力模块，用深度可分离卷积替换普通卷积
    """
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br):
        """
        定义一个全局的极化多尺度特征自注意力模块，用深度可分离卷积替换普通卷积

        :param in_channels: 输入各尺度特征图的通道数
        :param max_pool_kernels: 输入各尺度特征图的下采样核大小
        :param ch: 全局特征统一的通道数
        :param ch_k: K的通道数
        :param ch_v: V的通道数
        :param br: 多尺度特征的数量
        """
        super(GlobalPMFSBlock_AP_Separate, self).__init__()
        # 初始化参数
        self.ch_bottle = in_channels[-1]
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_in = self.ch * self.br

        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            DepthWiseSeparateConvBlock(
                in_channel=in_channel,
                out_channel=self.ch,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True
            )
            for in_channel in in_channels
        ])

        # 定义将全局特征下采样到相同尺寸的最大池化层
        self.max_pool_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=k)
            for k in max_pool_kernels
        ])

        # 定义通道Wq卷积
        self.ch_Wq = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True)
        # 定义通道Wk卷积
        self.ch_Wk = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=1, kernel_size=1, stride=1, batch_norm=True, preactivation=True)
        # 定义通道Wv卷积
        self.ch_Wv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True)
        # 定义通道K的softmax
        self.ch_softmax = nn.Softmax(dim=1)
        # 定义对通道分数矩阵的卷积
        self.ch_score_conv = nn.Conv2d(self.ch_in, self.ch_in, 1)
        # 定义对通道分数矩阵的LayerNorm层归一化
        self.ch_layer_norm = nn.LayerNorm((self.ch_in, 1, 1))
        # 定义sigmoid
        self.sigmoid = nn.Sigmoid()

        # 定义空间Wq卷积
        self.sp_Wq = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1, batch_norm=True, preactivation=True)
        # 定义空间Wk卷积
        self.sp_Wk = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1, batch_norm=True, preactivation=True)
        # 定义空间Wv卷积
        self.sp_Wv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_v, kernel_size=1, stride=1, batch_norm=True, preactivation=True)
        # 定义空间K的softmax
        self.sp_softmax = nn.Softmax(dim=-1)
        # 定义空间卷积，还原通道数
        self.sp_output_conv = DepthWiseSeparateConvBlock(in_channel=self.br * self.ch_v, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True)

        # 定义输出卷积，将通道数转换为输入的瓶颈层特征图通道数
        self.output_conv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_bottle, kernel_size=3, stride=1, batch_norm=True, preactivation=True)

    def forward(self, feature_maps):
        # 用最大池化统一特征图尺寸
        max_pool_maps = [
            max_pool_layer(feature_maps[i])
            for i, max_pool_layer in enumerate(self.max_pool_layers)
        ]
        # 计算通道各分支
        ch_outs = [
            ch_conv(max_pool_maps[i])
            for i, ch_conv in enumerate(self.ch_convs)
        ]
        # 将不同分支的特征进行拼接
        x = torch.cat(ch_outs, dim=1)  # bs, self.ch_in, h, w
        # 获得拼接后特征图的维度信息
        bs, c, h, w = x.size()

        # 先计算通道ch_Q、ch_K、ch_V
        ch_Q = self.ch_Wq(x)  # bs, self.ch_in, h, w
        ch_K = self.ch_Wk(x)  # bs, 1, h, w
        ch_V = self.ch_Wv(x)  # bs, self.ch_in, h, w
        # 转换通道ch_Q维度
        ch_Q = ch_Q.reshape(bs, -1, h * w)  # bs, self.ch_in, h*w
        # 转换通道ch_K维度
        ch_K = ch_K.reshape(bs, -1, 1)  # bs, h*w, 1
        # 对通道ch_K采取softmax
        ch_K = self.ch_softmax(ch_K)  # bs, h*w, 1
        # 将通道ch_Q和通道ch_K相乘
        Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1)  # bs, self.ch_in, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1
        # 通道增强
        ch_out = ch_V * ch_score  # bs, self.ch_in, h, w

        # 先计算空间sp_Q、sp_K、sp_V
        sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, h, w
        sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, h, w
        sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, h, w
        # 转换空间sp_Q维度
        sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, h*w*self.br
        # 转换空间sp_K维度
        sp_K = sp_K.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
        # 转换空间sp_V维度
        sp_V = sp_V.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)  # bs, self.ch_v, h, w, self.br
        # 对空间sp_K采取softmax
        sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
        # 将空间sp_K和空间sp_Q相乘
        Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, h, w, self.br)  # bs, 1, h, w, self.br
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(Z)  # bs, 1, h, w, self.br
        # 空间增强
        sp_out = sp_V * sp_score  # bs, self.ch_v, h, w, self.br
        # 变换空间增强后的维度
        sp_out = sp_out.permute(0, 4, 1, 2, 3).reshape(bs, self.br * self.ch_v, h, w)  # bs, self.br*self.ch_v, h, w
        # 还原通道数
        sp_out = self.sp_output_conv(sp_out)  # bs, self.ch_in, h, w

        # 最终的输出卷积，将通道数转换为输入的瓶颈层特征图通道数
        out = self.output_conv(sp_out)

        return out





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = [
        torch.randn((1, 32, 80, 80)).to(device),
        torch.randn((1, 64, 40, 40)).to(device),
        torch.randn((1, 128, 20, 20)).to(device),
    ]

    model = GlobalPMFSBlock_AP([32, 64, 128], [4, 2, 1], 64, 64, 64, 3).to(device)

    output = model(x)

    print(output.size())










