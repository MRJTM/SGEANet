"""
make geometry adpative ground truth

"""
import os
import sys
import cv2
import numpy as np
import scipy.io as io
sys.path.append(os.getcwd())
from data.prepare_dataset.density_map_utils import generate_ga_density_map

"""--------------------config--------------------"""
data_root_path='data/SHA'
data_name_list=['train','test']

"""--------------------Processing---------------"""
# process image one by one
for data_name in data_name_list:
    print("-------------------[{}]-------------------".format(data_name))
    image_folder_path=os.path.join(data_root_path,data_name,'images')
    gt_folder_path=os.path.join(data_root_path,data_name,'npy_ga_gt')
    if not os.path.exists(gt_folder_path):
        os.mkdir(gt_folder_path)
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

        # generate ga_density_map
        point_map=np.zeros((img.shape[0],img.shape[1]))
        for i in range(0, len(point_gt)):
            if int(point_gt[i][1]) < img.shape[0] and int(point_gt[i][0]) < img.shape[1]:
                point_map[int(point_gt[i][1]), int(point_gt[i][0])] = 1
        density_map=generate_ga_density_map(point_map,max_size=48,dis_sigma_rate=7)

        # save density map
        gt_path = os.path.join(gt_folder_path, img_name.replace('jpg', 'npy'))
        np.save(gt_path,density_map)

