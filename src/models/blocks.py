"""
@File       : blocks.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2019/10/5
@Desc       :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import sys
sys.path.append('../..')

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        pad=dilation*(kernel-1)//2
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=pad,dilation=dilation)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input

class BaseDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, activation=None, use_bn=False):
        super(BaseDeConv, self).__init__()
        pad = kernel // 2
        if dilation > 1:
            pad = dilation
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding=pad,dilation=dilation)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input

class Conv_block(nn.Module):
    def __init__(self,layers,in_channels,out_channels,ksize=3,stride=1,BN=False,dilation=1):
        super(Conv_block,self).__init__()
        pad=ksize//2
        if dilation>1:
            pad=dilation
        # 先创建第一层
        block=[nn.Conv2d(in_channels,out_channels,ksize,stride,padding=pad,dilation=dilation)]
        if BN:
            block+=[nn.BatchNorm2d(out_channels)]
        block+=[nn.ReLU(inplace=True)]

        # 然后创建接下来几层
        conv2d = nn.Conv2d(out_channels, out_channels, ksize, stride,padding=pad,dilation=dilation)
        if layers>1:
            for i in range(layers-1):
                block+=[conv2d]
                if BN:
                    block += [nn.BatchNorm2d(out_channels)]
                block += [nn.ReLU(inplace=True)]

        self.model=nn.Sequential(*block)


    def forward(self, x):
        output=self.model(x)
        return output

class Density_Predictor(nn.Module):
    def __init__(self,in_channels):
        super(Density_Predictor,self).__init__()
        self.Conv1=Conv_block(layers=1,in_channels=in_channels,out_channels=128,ksize=3)
        self.Conv2=Conv_block(layers=1,in_channels=128,out_channels=64,ksize=3)
        self.Conv3=nn.Conv2d(64,1,1,padding=0)

    def forward(self, x):
        x=self.Conv1(x)
        x=self.Conv2(x)
        x=self.Conv3(x)
        return x









