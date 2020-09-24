"""
@File       : src.py
@Author     : caozhijie
@Email      : 
@Date       : 2019/10/5
@Desc       :
"""

import torch
import shutil
import os

# save checkppint
def save_checkpoint(state, is_best, model_save_folder_path,model_name=None):
    if model_name:
        checkpoint_path=os.path.join(model_save_folder_path,'{}_checkpoint.pth.tar'.format(model_name))
        best_model_path=os.path.join(model_save_folder_path,'{}_best.pth.tar'.format(model_name))
    else:
        checkpoint_path = os.path.join(model_save_folder_path, 'checkpoint.pth.tar'.format(model_name))
        best_model_path = os.path.join(model_save_folder_path, 'best.pth.tar'.format(model_name))
    # save checkpoint
    torch.save(state, checkpoint_path)
    # save best model if better
    if is_best:
        shutil.copyfile(checkpoint_path,best_model_path)

# load checkpoint
def load_checkpoint(model,checkpoint_path):
    print("[INFO] load pretrain model")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    best_epoch = checkpoint['epoch']
    best_mae = checkpoint['best_MAE']
    best_mse = checkpoint['best_MSE']

    print("[MODEL INFO]:")
    print("MAE:",best_mae)
    print("MSE:",best_mse)
    print("EPOCH:",best_epoch)
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    return model,best_epoch,best_mae,best_mse,checkpoint
