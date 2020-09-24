"""
@File       : make_ga_synthetic.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/25
@Desc       :
"""

import os
import sys
import cv2
import numpy as np
import scipy.io as io
sys.path.append(os.getcwd())
from data.prepare_dataset.synthetic_utils import generate_ga_syn_image

"""--------------------config--------------------"""
data_root_path='data/SHB'
data_name_list=['train','test']

"""--------------------Processing---------------"""
# process image one by one
for data_name in data_name_list:
    print("-------------------[{}]-------------------".format(data_name))
    image_folder_path=os.path.join(data_root_path,data_name,'images')
    syn_folder_path=os.path.join(data_root_path,data_name,'synthetic_ga')
    if not os.path.exists(syn_folder_path):
        os.mkdir(syn_folder_path)
    image_names=os.listdir(image_folder_path)
    image_names.sort()
    print("image num=",len(image_names))

    for i,img_name in enumerate(image_names):
        print("[{}/{}]:{}-{}".format(i+1,len(image_names),data_name,img_name))
        img_path=os.path.join(image_folder_path,img_name)
        point_gt_path=img_path.replace('images','point_gt').replace('jpg','mat').replace('IMG_','GT_IMG_')

        # load img
        img = cv2.imread(img_path)

        # load point_gt
        mat=io.loadmat(point_gt_path)
        point_gt=mat["image_info"][0,0][0,0][0]

        # generate sythetic_ga
        point_map=np.zeros((img.shape[0],img.shape[1]))
        for i in range(0, len(point_gt)):
            if int(point_gt[i][1]) < img.shape[0] and int(point_gt[i][0]) < img.shape[1]:
                point_map[int(point_gt[i][1]), int(point_gt[i][0])] = 1
        syn_img=generate_ga_syn_image(point_map,max_size=48)

        # save density map
        syn_save_path = os.path.join(syn_folder_path, img_name)
        cv2.imwrite(syn_save_path,syn_img)