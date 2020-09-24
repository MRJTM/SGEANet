"""
@File       : SG_train.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/31
@Desc       :
"""

import os
import sys
import numpy as np
import copy

sys.path.append(os.getcwd())
from src.utils.common_utils import create_progressbar

def train(real_model, syn_model, cfg, train_loader, loss_zoo, en_optimizer,de_optimizer, iter=0, viz=None,
          loss_win_list=[],time_index=""):
    real_model.train()
    syn_model.eval()
    pbar = create_progressbar("Train", len(train_loader))
    # train batch by batch

    # 打开一系列loss记录文档
    loss_file_dict = {}
    for loss_name in cfg['loss_list']:
        loss_log_file_name = 'log_loss_{}_{}.txt'.format(loss_name, time_index)
        loss_log_file_path = os.path.join(cfg['log_folder_path'], loss_log_file_name)
        loss_file_dict[loss_name] = open(loss_log_file_path, 'a')

    for i, (img, syn, gt, img_path) in enumerate(train_loader):
        # print("img_path:",img_path)
        img = img.cuda(device=cfg['gpu_id'])
        syn = syn.cuda(device=cfg['gpu_id'])
        gt = gt.cuda(device=cfg['gpu_id'])
        gt = gt * cfg['gt_enlarge_rate']

        # compute content loss
        if 'content_loss' in cfg['loss_list']:
            index=cfg['loss_list'].index('content_loss')
            loss_weight=cfg['loss_weight_list'][index]
            fs2, fs3, fs4, fs5 = syn_model.forward(syn, mode=cfg['feature_mode'])
            fr2, fr3, fr4, fr5 = real_model.forward(img, mode=cfg['feature_mode'])
            ct_loss2 = loss_zoo('content_loss',cfg)(fr2, fs2.detach())
            ct_loss3 = loss_zoo('content_loss',cfg)(fr3, fs3.detach())
            ct_loss4 = loss_zoo('content_loss',cfg)(fr4, fs4.detach())
            ct_loss5 = loss_zoo('content_loss',cfg)(fr5, fs5.detach())
            # print("ct_loss1:", ct_loss1)
            # print("ct_loss2:", ct_loss2)
            # print("ct_loss3:", ct_loss3)
            # print("ct_loss4:", ct_loss4)
            loss = loss_weight[0] * ct_loss2 + loss_weight[1] * ct_loss3 + \
                   loss_weight[2] * ct_loss4 + loss_weight[3] * ct_loss5
            en_optimizer.zero_grad()
            loss.backward()
            en_optimizer.step()
            if viz:
                x = np.array([iter])
                y = np.array([loss.item() / cfg['batch_size']])
                viz.line(X=x, Y=y, win=loss_win_list[index], update='append')

            loss_file_dict['content_loss'].writelines("{},{}\n".format(iter,loss.item()))

        # compute other losses
        have_other_loss=False
        density_map = real_model.forward(img, mode='dmp')
        total_dmp_loss=0
        for j,loss_name in enumerate(cfg['loss_list']):
            if loss_name!='content_loss':
                loss=loss_zoo(loss_name,cfg)(density_map,gt)
                weight=cfg['loss_weight_list'][j]
                # print("{}:{},weight={}".format(loss_name,loss,weight))
                total_dmp_loss+=loss*weight
                have_other_loss=True

                if viz:
                    x = np.array([iter])
                    y = np.array([loss.item() / cfg['batch_size']])
                    viz.line(X=x, Y=y, win=loss_win_list[j], update='append')

                loss_file_dict[loss_name].writelines("{},{}\n".format(iter, loss.item()))
        if have_other_loss:
            de_optimizer.zero_grad()
            total_dmp_loss.backward()
            de_optimizer.step()

        iter += 1
        pbar.update(i)

    pbar.finish()

    # 关闭log文档
    for loss_name in cfg['loss_list']:
        loss_file_dict[loss_name].close()

    return real_model,syn_model, en_optimizer,de_optimizer, iter
