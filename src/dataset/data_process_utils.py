"""
@File       : data_process_utils.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/31
@Desc       :
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import math

# load data,just load data, no preprocess
def load_data(img_path,image_folder_name='images',gt_folder_name='ga_gt',syn_folder_name="synthetic_ga",train=False,dataset_type='base'):
    # load img
    img=cv2.imread(img_path.replace('images', image_folder_name))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # load gt
    if train:
        if gt_folder_name[:3]=='npy':
            gt_path = img_path.replace('images', gt_folder_name).replace('jpg', 'npy')
            gt=np.load(gt_path)
        else:
            gt_path=img_path.replace('images',gt_folder_name).replace('jpg','h5')
            gt = np.array(h5py.File(gt_path, 'r')['density']).astype(np.float32)
    else:
        gt_path = img_path.replace('images','gt_num').replace('jpg', 'npy')
        gt = np.load(gt_path)

    # if not base, load synthetic
    if dataset_type=='SG' and train:
        syn_path=img_path.replace('images',syn_folder_name)
        syn = cv2.imread(syn_path)
        syn = cv2.cvtColor(syn, cv2.COLOR_BGR2RGB)
    else:
        syn=None

    if dataset_type=='SG':
        return img,syn,gt
    else:
        return img,gt

# get target size,to make the width and height all 32 times
# returned target size is (w,h)
def get_target_size(img,short_size=512):
    h=img.shape[0]
    w=img.shape[1]
    if min(h,w)<short_size:
        if h<w:
            target_h=512
            target_w=int(w*(512/h)//32*32)
        else:
            target_w=512
            target_h=int(h*(512/w)//32*32)
    else:
        target_w=int(w//32*32)
        target_h=int(h//32*32)

    return (target_w,target_h)

def resize_gt(gt,h,w):
    resized_gt=cv2.resize(gt,(w,h))
    gt_num_before_resize=np.sum(gt)
    gt_num_after_resize=np.sum(resized_gt)
    if gt_num_before_resize==0 or gt_num_after_resize==0:
        gt_change_rate=1
    else:
        gt_change_rate=gt_num_before_resize/gt_num_after_resize
    resized_gt=resized_gt*gt_change_rate
    return resized_gt

# same crop
# crop_size=(w,h)
def get_crop_start_point(H,W,crop_H,crop_W):
    if W-crop_W>0:
        start_x = np.random.randint(0, W - crop_W)
    else:
        start_x=0
    if H-crop_H>0:
        start_y = np.random.randint(0, H - crop_H)
    else:
        start_y=0
    return start_x,start_y

# same flip
def same_flip(img,gt):
    img=cv2.flip(img,1)
    gt=cv2.flip(gt,1)
    return img,gt


# to prevent the conflict of channel concat because different size
def fix_singular_shape(img, unit_len=16):
    hei_dst, wid_dst = img.shape[0] + (unit_len - img.shape[0] % unit_len), img.shape[1] + (unit_len - img.shape[1] % unit_len)
    if len(img.shape) == 3:
        img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
    elif len(img.shape) == 2:
        sum_before_resize = max(int(round(np.sum(img))),1)
        img = cv2.resize(img, (wid_dst, hei_dst), interpolation=cv2.INTER_LANCZOS4)
        sum_after_resize=max(int(round(np.sum(img))),1)
        img = img*(sum_before_resize/sum_after_resize)
    return img