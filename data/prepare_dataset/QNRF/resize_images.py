"""
@File       : resize_images.py
@Author     : caozhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/2/5
@Desc       : QNRF需要resize到1280*720，所以事先将image resize好
"""
import os
import cv2

"""--------------------config---------------------"""
root='/media/Disk1/caozhijie/dataset'
dataset_name='QNRF'
train_or_test='test'
data_root_path=os.path.join(root,dataset_name,'{}_data'.format(train_or_test))

image_folder_path=os.path.join(data_root_path,'images')
resized_image_folder_path=os.path.join(data_root_path,'resized_images_720P')

if not os.path.exists(resized_image_folder_path):
    os.mkdir(resized_image_folder_path)

image_names=os.listdir(image_folder_path)
image_names.sort()

print("[INFO]: Processing {}-{}".format(dataset_name,train_or_test))
print("data_num:",len(image_names))

"""--------------------processing-----------------"""
for i,img_name in enumerate(image_names):
    print("[{}/{}]: {}".format(i+1,len(image_names),img_name))
    img_path=os.path.join(image_folder_path,img_name)

    # load img
    img=cv2.imread(img_path)

    # resize img
    resized_img=cv2.resize(img,(1280,720))

    # save resized_img
    resized_img_save_path=os.path.join(resized_image_folder_path,img_name)
    cv2.imwrite(resized_img_save_path,resized_img)


