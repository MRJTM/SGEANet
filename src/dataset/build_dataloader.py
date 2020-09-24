"""
@File       : build_dataloader.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/26
@Desc       :
"""

import os
import sys
import json
sys.path.append(os.getcwd())
import torch
from src.dataset.dataset_base import baseDataset
from src.dataset.dataset_SG import SGDataset
from torchvision import transforms

def base_dataloader(cfg,train=True):
    # config transform->dataset->dataloader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    if train:
        with open(cfg['train_list_path'], 'r') as f:
            img_paths = json.load(f)
    else:
        with open(cfg['test_list_path'], 'r') as f:
            img_paths = json.load(f)

    dataset = baseDataset(image_paths=img_paths,
                           transform=transform,
                           train=train,
                           cfg=cfg,
                           data_copy_num=1)

    dataloader = torch.utils.data.DataLoader(dataset,
                                               batch_size=cfg['batch_size'] if train else 1,
                                               shuffle=train)

    return dataloader

def SG_dataloader(cfg,train=True):
    # config transform->dataset->dataloader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    if train:
        with open(cfg['train_list_path'], 'r') as f:
            img_paths = json.load(f)
    else:
        with open(cfg['test_list_path'], 'r') as f:
            img_paths = json.load(f)

    dataset = SGDataset(image_paths=img_paths,
                          transform=transform,
                          train=train,
                          cfg=cfg,
                          data_copy_num=1)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg['batch_size'] if train else 1,
                                             shuffle=train)

    return dataloader
