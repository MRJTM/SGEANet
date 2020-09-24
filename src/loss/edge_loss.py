"""
@File       : edge_loss.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/31
@Desc       : loss of edges
"""

import cv2
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.nn import BCELoss

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    kernel /= kernel.sum()
    return kernel

# compute only on edge
class base_edge_loss(_Loss):
    def __init__(self, sharp=100,gpu_id=0):
        super(base_edge_loss, self).__init__()
        self.sharp=sharp
        self.gpu_id=gpu_id

    def forward(self, pred, gt):
        n,c,h,w=pred.shape

        # get gt edge
        instance_mask = torch.zeros((n, c, h, w))
        instance_mask[torch.where(gt > 0)] = 1
        instance_mask=instance_mask.cuda(device=self.gpu_id)

        gt_edge=1-gt
        gt_edge = gt_edge ** self.sharp
        gt_edge = gt_edge.mul(instance_mask)

        pred_edge=1-pred
        pred_edge = pred_edge ** self.sharp
        pred_edge = pred_edge.mul(instance_mask)

        loss=torch.sum(torch.log(torch.pow(pred_edge-gt_edge,2)+1))

        return loss


class multi_scale_edge_loss(_Loss):
    def __init__(self, sharp=100,gpu_id=0):
        super(multi_scale_edge_loss, self).__init__()
        self.sharp=sharp
        self.gpu_id=gpu_id

    def forward(self, pred, gt):
        n,c,h,w=pred.shape

        # get gt edge
        instance_mask = torch.zeros((n, c, h, w))
        instance_mask[torch.where(gt > 0)] = 1
        instance_mask[torch.where(gt>1)]=0
        instance_mask[torch.where(pred>1)]=0
        instance_mask[torch.where(pred<0)]=0
        instance_mask=instance_mask.cuda(device=self.gpu_id)

        gt_edge=1-gt
        gt_edge = gt_edge.mul(instance_mask)
        gt_edge = gt_edge ** self.sharp

        pred_edge=1-pred
        pred_edge = pred_edge.mul(instance_mask)
        pred_edge = pred_edge ** self.sharp
        loss1 = torch.sum(torch.log(torch.pow(pred_edge - gt_edge, 2) + 1))
        # print("loss1:",loss1)

        # conv ksize=5, sigma=1.1
        kernel=gaussian_kernel(5,1.1)
        weight = np.tile(kernel, (1, 1, 1, 1))
        weight = Parameter(torch.from_numpy(weight).float(), requires_grad=False).cuda(device=self.gpu_id)
        pred_edge2 = F.conv2d(pred_edge, weight, padding=2)
        gt_edge2= F.conv2d(gt_edge, weight, padding=2)
        loss2=torch.sum(torch.log(torch.pow(pred_edge2-gt_edge2,2)+1))
        # print("loss2:", loss2)

        # conv ksize=9,sigma=1.7
        kernel = gaussian_kernel(9, 1.7)
        weight = np.tile(kernel, (1, 1, 1, 1))
        weight = Parameter(torch.from_numpy(weight).float(), requires_grad=False).cuda(device=self.gpu_id)
        pred_edge3 = F.conv2d(pred_edge, weight, padding=4)
        gt_edge3 = F.conv2d(gt_edge, weight, padding=4)
        loss3 = torch.sum(torch.log(torch.pow(pred_edge3 - gt_edge3, 2) + 1))
        # print("loss3:", loss3)

        # conv ksize=15,sigma=2.6
        kernel = gaussian_kernel(15, 2.6)
        weight = np.tile(kernel, (1, 1, 1, 1))
        weight = Parameter(torch.from_numpy(weight).float(), requires_grad=False).cuda(device=self.gpu_id)
        pred_edge4 = F.conv2d(pred_edge, weight, padding=7)
        gt_edge4 = F.conv2d(gt_edge, weight, padding=7)
        loss4 = torch.sum(torch.log(torch.pow(pred_edge4 - gt_edge4, 2) + 1))
        # print("loss4:", loss4)

        # enrode
        gt_edge = 1 - gt
        gt_edge = gt_edge.mul(instance_mask)
        gt_edge = gt_edge ** 300

        pred_edge = 1 - pred
        pred_edge = pred_edge.mul(instance_mask)
        pred_edge = pred_edge ** 300
        loss5 = torch.sum(torch.log(torch.pow(pred_edge - gt_edge, 2) + 1))
        # print("loss5:", loss5)

        # dilate
        gt_edge = 1 - gt
        gt_edge = gt_edge.mul(instance_mask)
        gt_edge = gt_edge ** 20


        pred_edge = 1 - pred
        pred_edge = pred_edge.mul(instance_mask)
        pred_edge = pred_edge ** 20
        loss6 = torch.sum(torch.log(torch.pow(pred_edge - gt_edge, 2) + 1))
        # print("loss6:", loss6)

        loss=loss1+loss2+loss3+loss4+loss5+loss6
        return loss