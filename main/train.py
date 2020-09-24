"""
@File       : train.py
@Author     : caozhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2019/10/5
@Desc       : This is to train one model
"""

# python origin packages
import os
import sys
import copy
import time
import numpy as np
from visdom import Visdom

# self-define packages
sys.path.append(os.getcwd())
sys.path.append('.')
sys.path.append('../..')
from src.zoo import *
from src.config.config import *
from src.utils.model import *

# torch packages
import torch

"""-------------------------get config from config file-------------------------"""
# parse config file
cfg=parse_configs(sys.argv[1],train_or_test='train')
print_config(cfg)

"""---------------------config dataloader-------------------"""
print("\n------- 1.load data -------")
build_dataloader=dataloader_zoo[cfg['data_type']]
train_loader=build_dataloader(cfg,train=True)
val_loader=build_dataloader(cfg,train=False)

"""-----------------2. config model-----------------------"""
print("\n------------2.config model----------")
# model structure
model=model_zoo[cfg['model_name']]
model.cuda(device=cfg['gpu_id'])
if cfg['train_type']=='SG':
    tmp_model = model_zoo[cfg['model_name']]
    syn_model = copy.deepcopy(tmp_model)
    syn_model.cuda(device=cfg['gpu_id'])
print("model is {}".format(cfg['model_name']))
# summary(model,(3,512,512))

# load pretrain model
if (not cfg['train_from_scratch']) and os.path.exists(cfg['pretrain_path']):
    model,best_epoch,best_mae,best_mse,checkpoint=load_checkpoint(model,cfg['pretrain_path'])
    start_epoch = best_epoch + 1
else:
    best_mae = 1e10
    best_mse = 1e10
    best_epoch = 0
    start_epoch = 1

# load syn model
if cfg['train_type']=='SG':
    if os.path.exists(cfg['syn_pretrain_path']):
        syn_model,_,_,_,syn_checkpoint=load_checkpoint(syn_model,cfg['syn_pretrain_path'])
        # print('copy syn_model.decoder to real_model.decoder')
        # print("syn_checkpoint:",syn_checkpoint['state_dict'])
        real_model_dict = model.state_dict()
        state_dict = {k: v for k, v in syn_checkpoint['state_dict'].items() if k.split('.')[0]=='decoder'}
        # print("\nreal_dict:",state_dict)  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        real_model_dict.update(state_dict)
        model.load_state_dict(real_model_dict)
    else:
        print("can not find the syn_model from {}".format(cfg['syn_pretrain_path']))
        exit(-1)

# optimizer
if cfg['train_type']=='SG':
    en_optimizer, de_optimizer = optim_zoo_SG(model, cfg)
else:
    optimizer = optim_zoo(model, cfg)

"""-------------------3. train--------------------------"""
print("\n-----------3.train-----------")
iter=0
time_index=time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
# create loss curve window
env_name="{}-{}-{}".format(cfg['dataset_name'],cfg['model_name'],cfg['model_save_name'])
viz = Visdom(env=env_name)
assert viz.check_connection()

loss_win_list=[]
for loss_name in cfg['loss_list']:
    win=viz.line(X=torch.zeros((1,)).cpu(),Y=torch.zeros((1,)).cpu(),
        opts=dict(xlabel="iter",ylabel="{}".format(loss_name),
                  title="{} vs iter".format(loss_name)))
    loss_win_list.append(win)

MAE_win=viz.line(X=torch.ones((1,)).cpu(),Y=torch.zeros((1,)).cpu(),
        opts=dict(xlabel="epoch",ylabel="MAE",title="MAE vs epoch"))

train_func=train_func_zoo[cfg['train_type']]
test_func=test_func_zoo[cfg['test_type']]

MAE_log_file_path=os.path.join(cfg['log_folder_path'],'log_metric_MAE_{}.txt'.format(time_index))
MSE_log_file_path=os.path.join(cfg['log_folder_path'],'log_metric_MSE_{}.txt'.format(time_index))
for epoch_index in range(start_epoch,cfg['epoch']):
    print("[Epoch {}/{}]".format(epoch_index,cfg['epoch']))
    """Train on train data"""
    if cfg['train_type']=='SG':
        model, syn_model,en_optimizer, de_optimizer, iter = train_func(model, syn_model, cfg, train_loader, loss_zoo,
                                                                  en_optimizer, de_optimizer, iter, viz,
                                                                  loss_win_list,time_index)
    else:
        model,optimizer,iter=train_func(model,cfg,train_loader,loss_zoo,optimizer,iter,viz,loss_win_list,time_index)

    """Val on test data"""
    val_result=test_func(model,cfg,val_loader,train=True)
    mae=val_result['MAE']
    mse=val_result['MSE']
    MAE_log_file = open(MAE_log_file_path, 'a')
    MSE_log_file = open(MSE_log_file_path, 'a')
    MAE_log_file.writelines("{},{}\n".format(epoch_index, mae))
    MSE_log_file.writelines("{},{}\n".format(epoch_index, mse))
    MAE_log_file.close()
    MSE_log_file.close()

    # update MAE curve
    viz.line(X=np.array([epoch_index]),Y=np.array([mae]),win=MAE_win,update='append')

    is_best = False
    if mae < best_mae:
        best_mae = mae
        best_mse = mse
        best_epoch = epoch_index
        is_best = True

    print("MAE={:.2f}| [Best]:MAE={:.2f}, MSE:{:.2f},epoch={}".format(
        mae, best_mae, best_mse, best_epoch
    ))
    """save better model"""
    save_checkpoint({
        'state_dict': model.state_dict(),
        'epoch': epoch_index,
        'best_MAE': best_mae,
        'best_MSE': best_mse,
    }, is_best=is_best,
        model_save_folder_path=cfg['model_save_path'])


MAE_log_file.close()
MSE_log_file.close()