"""
@File       : SFANet1.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/21
@Desc       : modified SFANet, removed segmentation branch,
            : enlarged the output resolution to the same size with input image
"""

import torch
from torch import nn
from torch.utils import model_zoo
from src.models.blocks import BaseConv
from src.models.VGG16 import VGG16_backbone


class SGEANet(nn.Module):
    def __init__(self):
        super(SGEANet, self).__init__()
        self.vgg = VGG16_backbone()
        self.load_vgg()
        self.dmp = BackEnd()
        self.decoder = nn.Sequential(
            BaseConv(64, 64, 3, activation=nn.ReLU(),use_bn=False),
            BaseConv(64, 32, 3, activation=nn.ReLU(), use_bn=False),
            BaseConv(32, 1, 1, 1, activation=None, use_bn=False)
        )

    def forward(self, input,mode="dmp"):
        if mode=='decoder':
            dmp_out=self.decoder(input)
        else:
            input = self.vgg(input)
            conv2_2,conv3_3,conv4_3,conv5_3,x2,x3,x4=self.dmp(*input)
            dmp_out = self.decoder(x2)

        if mode=='feature_vgg':
            return conv2_2,conv3_3,conv4_3,conv5_3
        elif mode=='feature_dmp':
            return x2,x3,x4,conv5_3
        elif mode=='feature_and_dmp':
            return x2,x3,x4,conv5_3,dmp_out
        else:
            return dmp_out

    # load pretrained vgg weights
    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)



    def forward(self, *input):
        conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)          # 1/16->1/8

        input = torch.cat([input, conv4_3], 1)  # 1/8:512+512=1024
        input = self.conv1(input)               # 1024->256
        x4 = self.conv2(input)               # 256->256

        input = self.upsample(x4)            # 1/8->1/4
        input = torch.cat([input, conv3_3], 1)  # 1/4:256+256=512
        input = self.conv3(input)               # 512->128
        x3 = self.conv4(input)               # 128->128

        input = self.upsample(x3)            # 1/4->1/2
        input = torch.cat([input, conv2_2], 1)  # 1/2: 128+128=256
        input = self.conv5(input)               # 256->64
        x2 = self.conv6(input)               # 64->64

        return conv2_2,conv3_3,conv4_3,conv5_3,x2,x3,x4


