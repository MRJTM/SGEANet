"""
@File       : val.py
@Author     : caozhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2019/10/5
@Desc       :
"""

# python origin packages
import os
import sys
import time

# self-define packages
sys.path.append(os.getcwd())
sys.path.append('.')
sys.path.append('../..')
from src.zoo import *
from src.config.config import *
from src.utils.model import *

"""-------------------------get config from config file-------------------------"""
# parse config file
cfg=parse_configs(sys.argv[1],train_or_test='test')
print_config(cfg)

"""---------------------config dataloader-------------------"""
print("\n------- 1.load data -------")
build_dataloader=dataloader_zoo[cfg['data_type']]
val_loader=build_dataloader(cfg,train=False)

"""-----------------2. config model-----------------------"""
print("\n------------2.config model----------")
# model structure
model=model_zoo[cfg['model_name']]
print("model is {}".format(cfg['model_name']))
model.cuda(device=cfg['gpu_id'])
# summary(model,(3,512,512))

# load pretrain model
if os.path.exists(cfg['pretrain_path']):
    model,best_epoch,best_mae,best_mse,checkpoint=load_checkpoint(model,cfg['pretrain_path'])
else:
    print("{} doesn't exist!!!".format(cfg['pretrain_path']))
    exit(0)

"""-------------------3. test--------------------------"""
print("\n-----------3.test-----------")
test_func=test_func_zoo[cfg['test_type']]

"""Val on test data"""
time_index=time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
val_result=test_func(model,cfg,val_loader,train=False,time_index=time_index)
print("best_epoch:",best_epoch)