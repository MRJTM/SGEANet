"""
@Proj       : crowdcounting_code
@Name       : dataset_utils.py
@Author     : caozhijie
@Data       : 2019/9/4      
@Email      : caozhijie1@sensetime.com
@Desc       : 
"""

import cv2
import numpy as np
import copy
import progressbar
import scipy
import scipy.ndimage
from scipy import spatial

def mask_with_roi(image,roi_point_list):
    masked_image=copy.deepcopy(image)

    roi_points = np.array(roi_point_list)
    roi_points = roi_points.reshape((-1, 1, 2))
    # compute ROI
    h=image.shape[0]
    w=image.shape[1]
    ROI = np.zeros((h,w,3),np.uint8)
    cv2.fillPoly(ROI, np.int32([roi_points]), (0, 0, 255))

    # mask image and density map accornding to ROI
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # print("i,j=",i,j)
            if (ROI[i][j][0] == 0 and ROI[i][j][1] == 0 and ROI[i][j][2] == 255):
                pass
            else:
                if len(masked_image.shape)==2:
                    masked_image[i,j]=0
                else:
                    masked_image[i,j,:]=0

    # return result
    return masked_image

# 将class gt转化为one-hot形式
def one_hot(class_gt,K=4):
    one_hot_gt=np.zeros((class_gt.shape[0],class_gt.shape[1],K),np.uint8)
    for i in range(class_gt.shape[0]):
        for j in range(class_gt.shape[1]):
            one_hot_gt[i,j,int(class_gt[i,j])]=1
    return one_hot_gt

# 解码one hot为1,2,3,4的形式
def parse_one_hot(one_hot_gt):
    K=one_hot_gt.shape[2]
    class_gt=np.zeros((one_hot_gt.shape[0],one_hot_gt.shape[1]),np.uint8)
    for i in range(class_gt.shape[0]):
        for j in range(class_gt.shape[1]):
            for k in range(K):
                if one_hot_gt[i,j,k]==1:
                    class_gt[i,j]=k
    return class_gt

