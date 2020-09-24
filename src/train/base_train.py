"""
@File       : base_train.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/31
@Desc       :
"""

import os
import sys
import torch
import numpy as np
sys.path.append(os.getcwd())
from src.utils.common_utils import create_progressbar

def train(model,cfg,train_loader, loss_zoo, optimizer,iter=0,viz=None,loss_win_list=[],time_index=""):
    model.train()
    pbar = create_progressbar("Train", len(train_loader))
    # train batch by batch

    # 打开一系列loss记录文档
    loss_file_dict = {}
    for loss_name in cfg['loss_list']:
        loss_log_file_name = 'log_loss_{}_{}.txt'.format(loss_name, time_index)
        loss_log_file_path = os.path.join(cfg['log_folder_path'], loss_log_file_name)
        loss_file_dict[loss_name] = open(loss_log_file_path, 'a')

    for i, (img, gt, img_path) in enumerate(train_loader):
        # print("img_path:",img_path)
        img = img.cuda(device=cfg['gpu_id'])
        density_map = model(img)

        # compute counting loss : MSE loss
        gt = gt.cuda(device=cfg['gpu_id'])
        gt = gt * cfg['gt_enlarge_rate']

        # compute loss
        total_loss=0
        for j,loss_name in enumerate(cfg['loss_list']):
            loss_func=loss_zoo(loss_name,cfg)
            loss=loss_func(density_map,gt).cuda(device=cfg['gpu_id'])
            if torch.sum(torch.isnan(loss))>0:
                print("nan showed up in loss {}".format(loss_name))
                print("img_path=",img_path)
                print("pred_num=",torch.sum(density_map))
                print("gt_num=",torch.sum(gt))
                continue
            # print("{}:{}".format(loss_name,loss))
            loss=cfg['loss_weight_list'][j]*loss
            total_loss+=loss
            if viz:
                x = np.array([iter])
                y = np.array([loss.item() / cfg['batch_size']])
                viz.line(X=x, Y=y, win=loss_win_list[j], update='append')
            # 写入log
            loss_file_dict[loss_name].writelines("{},{}\n".format(iter,loss.item()))

        # total_loss=counting_loss*1000
        if total_loss==0:
            pass
        else:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        iter += 1
        pbar.update(i)

    pbar.finish()

    # 关闭log文档
    for loss_name in cfg['loss_list']:
        loss_file_dict[loss_name].close()

    return model, optimizer,iter