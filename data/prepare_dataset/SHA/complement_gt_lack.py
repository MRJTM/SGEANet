"""
@Proj       : crowdcounting_code
@Name       : complement_gt_lack.py
@Author     : caozhijie
@Data       : 2019/8/23      
@Email      : caozhijie1@sensetime
@Desc       : 
"""

import os

"""-----------------------------config---------------------------"""
src_gt_folder_name='npy_ga_gt'
dst_gt_folder_name='npy_det_ga_gt'
print("copy files from {} to {}".format(src_gt_folder_name,dst_gt_folder_name))

part_name='A'
train_or_test='train'
# data_root_path='/mnt/lustre/caozhijie1/dataset/counting/' \
#                'ShanghaiTech/part_{}_final/{}_data'.format(part_name,train_or_test)
data_root_path='/data/counting/' \
               'ShanghaiTech/part_{}_final/{}_data'.format(part_name,train_or_test)

src_gt_path=os.path.join(data_root_path,src_gt_folder_name)
dst_gt_path=os.path.join(data_root_path,dst_gt_folder_name)

src_gt_file_names=os.listdir(src_gt_path)
src_gt_file_names.sort()

dst_gt_file_names=os.listdir(dst_gt_path)
dst_gt_file_names.sort()

"""-------------------------complement gt files--------------------------"""
lack_gt_list=[]
for src_gt_name in src_gt_file_names:
    if src_gt_name not in dst_gt_file_names:
        src_gt_file_path=os.path.join(src_gt_path,src_gt_name)
        dst_gt_file_path=os.path.join(dst_gt_path,src_gt_name)
        cmd="cp {} {}".format(src_gt_file_path,dst_gt_file_path)
        os.system(cmd)
        lack_gt_list.append(src_gt_name)

print("lack_gt num:",len(lack_gt_list))
print("lack gt list:\n",lack_gt_list)