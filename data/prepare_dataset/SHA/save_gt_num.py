"""
@File       : save_gt_num.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/31
@Desc       : save the gt num, for decrease the memory used in val
"""

import os
import sys
import numpy as np
import scipy.io as io
sys.path.append(os.getcwd())

"""--------------------config--------------------"""
data_root_path='data/SHA'
data_name_list=['train','test']

"""--------------------Processing---------------"""
# process image one by one
for data_name in data_name_list:
    print("-------------------[{}]-------------------".format(data_name))
    image_folder_path=os.path.join(data_root_path,data_name,'images')
    gt_num_folder_path=os.path.join(data_root_path,data_name,'gt_num')
    if not os.path.exists(gt_num_folder_path):
        os.mkdir(gt_num_folder_path)
    image_names=os.listdir(image_folder_path)
    image_names.sort()
    print("image num=",len(image_names))

    for i,img_name in enumerate(image_names):
        print("[{}/{}]:{}-{}".format(i+1,len(image_names),data_name,img_name))
        img_path=os.path.join(image_folder_path,img_name)
        point_gt_path=img_path.replace('images','point_gt').replace('jpg','mat').replace('IMG_','GT_IMG_')

        # load point_gt
        mat=io.loadmat(point_gt_path)
        point_gt=mat["image_info"][0,0][0,0][0]

        # get gt_num
        gt_num=np.int(len(point_gt))
        print("gt_num:",gt_num)

        # save density map
        gt_path = os.path.join(gt_num_folder_path, img_name.replace('jpg', 'npy'))
        np.save(gt_path,gt_num)
