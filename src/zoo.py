"""
@File       : zoo.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/31
@Desc       : contains all kinds of zoos
"""

import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.getcwd())

"""models"""
from src.models.CSRNet import CSRNet
from src.models.SGEANet import SGEANet

model_zoo={
    'CSRNet':CSRNet(load_weights=False),
    'SGEANet': SGEANet(),
}

"""dataloader"""
from src.dataset.build_dataloader import base_dataloader
from src.dataset.build_dataloader import SG_dataloader
dataloader_zoo={
    'base':base_dataloader,
    'SG':SG_dataloader,
}

"""train function"""
from src.train import base_train
from src.train import SG_train

train_func_zoo={
    'base':base_train.train,
    'SG':SG_train.train
}

"""test function"""
from src.test import base_test

test_func_zoo={
    'base':base_test.test
}

"""loss"""
from src.loss.edge_loss import base_edge_loss,multi_scale_edge_loss

def loss_zoo(loss_name,cfg):
    if loss_name=='mse':
        return nn.MSELoss(reduction='sum')
    elif loss_name=='base_edge_loss':
        return base_edge_loss(gpu_id=cfg['gpu_id'],sharp=cfg['sharp'])
    elif loss_name=='multi_scale_edge_loss':
        return multi_scale_edge_loss(gpu_id=cfg['gpu_id'])
    elif loss_name=='synthetic_guided':
        return nn.MSELoss(reduction='sum')
    else:
        print("error loss name")
        exit(0)
# loss_zoo={
#     'MSE_total':nn.MSELoss(size_average=False),
#     'MSE_ave':nn.MSELoss(size_average=True),
#     'SSIM':SSIM_Loss(size=5,sigma=1.5,size_average=False),
#     'EDGE_ave':edge_Loss(size_average=True,max_value=100),
#     'EDGE_total':edge_Loss(size_average=False,max_value=100),
#     'EDGE2':edge_Loss2(size_average=False,gpu_id=cfg['gpu_id']),
#     'content_loss':nn.MSELoss(size_average=False),
#     'MS_SSIM':MS_SSIM_Loss(size_average=False)
# }

"""optimizer"""
def optim_zoo(model,cfg):
    if cfg['optim']=='SGD':
        return torch.optim.SGD(model.parameters(),cfg['lr'],momentum=0.95,weight_decay=0.005)
    elif cfg['optim']=='Adam':
        return torch.optim.Adam(model.parameters(), cfg['lr'], weight_decay=0.005)
    else:
        print("No such optimizer!!!")
        exit(-1)

def optim_zoo_SG(model,cfg):
    if cfg['optim']=='SGD':
        if cfg['model_name']=='SGEANet':
            if cfg['feature_mode']=='feature_vgg':
                # encoder optimizer
                en_optim=torch.optim.SGD([
                    {'params': model.vgg.parameters()},
                ],cfg['lr'],momentum=0.95,weight_decay=0.005)
                # decoder
                de_optim=torch.optim.SGD([
                    {'params': model.dmp.parameters()},
                    {'params': model.decoder.parameters()}
                ],cfg['lr'],momentum=0.95,weight_decay=0.005)
            else:
                # encoder optimizer
                en_optim = torch.optim.SGD([
                    {'params': model.vgg.parameters()},
                    {'params': model.dmp.parameters()},
                ], cfg['lr'], momentum=0.95, weight_decay=0.005)
                # decoder
                de_optim = torch.optim.SGD([
                    {'params': model.dmp.parameters()},
                    {'params': model.decoder.parameters()}
                ], cfg['lr'], momentum=0.95, weight_decay=0.005)
        else:
            en_optim = torch.optim.SGD(model.parameters(), cfg['lr'], momentum=0.95, weight_decay=0.005)
            de_optim = torch.optim.SGD(model.parameters(), cfg['lr'], momentum=0.95, weight_decay=0.005)
        return en_optim,de_optim
    elif cfg['optim']=='Adam':
        if cfg['model_name'] == 'SGEANet':
            if cfg['feature_mode']=='feature_vgg':
                # encoder optimizer
                en_optim = torch.optim.Adam([
                    {'params': model.vgg.parameters()},
                ], cfg['lr'],  weight_decay=0.005)
                # decoder
                de_optim = torch.optim.Adam([
                    {'params': model.dmp.parameters()},
                    {'params': model.decoder.parameters()}
                ], cfg['lr'], weight_decay=0.005)
            else:
                # encoder optimizer
                en_optim = torch.optim.Adam([
                    {'params': model.vgg.parameters()},
                    {'params': model.dmp.parameters()},
                ], cfg['lr'], weight_decay=0.005)
                # decoder
                de_optim = torch.optim.Adam([
                    {'params': model.dmp.parameters()},
                    {'params': model.decoder.parameters()}
                ], cfg['lr'], weight_decay=0.005)
        else:
            en_optim = torch.optim.Adam(model.parameters(), cfg['lr'], weight_decay=0.005)
            de_optim = torch.optim.Adam(model.parameters(), cfg['lr'], weight_decay=0.005)

        return en_optim, de_optim
    else:
        print("No such optimizer!!!")
        exit(-1)
