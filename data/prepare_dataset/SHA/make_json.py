# encoding: utf-8
"""
此脚本用于产生包含图片路径的json文件，训练的时候读取json文件来进行训练
"""
import json
import os
import numpy.random as random

"""-------------------------划分测试集和验证集----------------------------"""
dataset_name='SHA'
data_root_path='data/{}'.format(dataset_name)
data_names=['train','test']

"""---------------------generating json file-----------------------"""
for data_name in data_names:
    print("----------------train----------------")
    image_folder_path=os.path.join(data_root_path,data_name,'images')
    image_names=os.listdir(image_folder_path)
    print("num of images:",len(image_names))
    image_paths=[]
    for img_name in image_names:
        img_path=os.path.join(image_folder_path,img_name)
        image_paths.append(img_path)

    """write to json file"""
    js = json.dumps(image_paths)
    json_file_path=os.path.join('data/list/{}_{}.json'.format(dataset_name,data_name))
    file = open(json_file_path, 'w')
    file.write(js)
    file.close()
    print("writed to:",json_file_path)

