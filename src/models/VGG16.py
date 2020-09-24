"""
@File       : VGG16.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2019/11/17
@Desc       : This is the VGG16 with FC removed
"""

import torch.nn as nn
from torchvision import models
from src.models.blocks import BaseConv


# VGG16 backbone
class VGG16_backbone(nn.Module):
    def __init__(self):
        super(VGG16_backbone, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        conv1_2 = self.conv1_2(input)   # 1:64
        input = self.pool(conv1_2)        # 1->1/2 downsample

        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)   # 1/2:128
        input = self.pool(conv2_2)      # 1/4 downsample

        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)   # 1/4:256
        input = self.pool(conv3_3)      # 1/8 downsample

        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)   # 1/8: 512
        input = self.pool(conv4_3)      # 1/16 downsample

        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)   # 1/16: 512

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3