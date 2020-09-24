"""
@File       : base_test.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/26
@Desc       :
"""

import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
from src.utils.common_utils import fix_str_len
from src.dataset.data_process_utils import resize_gt
from src.metric.metric_utils import compute_GAME
sys.path.append(os.getcwd())

def test(model,cfg,val_loader,train=False,time_index=""):
    model.eval()
    result={}
    part_result={}

    with torch.no_grad():
        for i, (img, counting_gt, img_path) in enumerate(val_loader):
            # predict
            img = img.cuda(device=cfg['gpu_id'])
            img = Variable(img,requires_grad = False)
            density_map=model.forward(img,mode='dmp')

            # require_grad false: prevent cuda out of memory
            density_map.detach_().cpu()
            img = img.cpu()

            # convert to numpy array
            density_map=density_map.cpu().numpy()

            # statistic
            pred_num = np.sum(density_map)
            pred_num = pred_num / cfg['gt_enlarge_rate']
            gt_num = np.float(counting_gt.numpy())
            AE = abs(pred_num - gt_num)
            SE = np.square(pred_num - gt_num)
            img_name = img_path[0].split(os.path.sep)[-1]

            if train:
                result['MAE']=result['MAE']+AE if 'MAE' in result.keys() else AE
                result['MSE'] = result['MSE'] + SE if 'MSE' in result.keys() else SE
                new_img_name=fix_str_len(img_name,fix_len=13)
                new_pred_num=fix_str_len('{:.3f}'.format(pred_num),fix_len=10)
                new_gt_num=fix_str_len('{}'.format(int(gt_num)),fix_len=5)
                print("[{}/{}]-{}: pred:{}, gt:{}, MAE={:.3f}".format(i+1,len(val_loader),new_img_name, new_pred_num, new_gt_num, AE))

            else:
                print("[{}/{}]:{}".format(i+1,len(val_loader),img_name))

                # if need to load density ground truth
                if cfg['GAME'] or cfg['PSNR'] or cfg['SSIM'] or cfg['save_prediction']:
                    pred_h = density_map.shape[2]
                    pred_w = density_map.shape[3]
                    pred_density_map = np.reshape(density_map, (pred_h, pred_w))
                    gt_density_map_path = img_path[0].replace('images', cfg['gt_folder_name']).replace('jpg', 'npy')
                    gt_density_map = np.load(gt_density_map_path)
                    gt_density_map = resize_gt(gt_density_map, pred_h, pred_w)

                if cfg['MAE']:
                    result['MAE'] = result['MAE'] + AE if 'MAE' in result.keys() else AE
                    print("AE:{:.3f}".format(AE))

                if cfg['MSE']:
                    result['MSE'] = result['MSE'] + SE if 'MSE' in result.keys() else SE
                    print("SE:{:.3f}".format(SE))

                if cfg['GAME']:
                    game=compute_GAME(pred_density_map,gt_density_map,4)
                    if 'GAME' not in result.keys():
                        result['GAME']=game
                    else:
                        for i in range(4):
                            result['GAME'][i]+=game[i]
                    print("GAME:",game)

                if cfg['PSNR']:
                    pass

                if cfg['SSIM']:
                    pass

                if cfg['save_prediction']:
                    pred_save_folder_path = os.path.join('result', cfg['dataset_name'], cfg['train_or_test'],
                                                         '{}_{}_{}'.format(cfg['dataset_name'],
                                                                           cfg['model_name'],
                                                                           cfg['model_save_name']))
                    if not os.path.exists(pred_save_folder_path):
                        os.makedirs(pred_save_folder_path)
                    img_name = img_path[0].split(os.path.sep)[-1]
                    pred_save_path = os.path.join(pred_save_folder_path, img_name.replace('jpg', 'npy'))
                    np.save(pred_save_path, pred_density_map)
                    print("saved pred_density_map")

                if cfg['partition']:
                    key=int(gt_num//100)*100
                    if key not in part_result.keys():
                        part_result[key]=[]
                    part_result[key].append(AE)

            torch.cuda.empty_cache()

    # average
    print("\n-----Final Statistic-----")
    for k in result.keys():
        if k=='MSE':
            result[k]=np.sqrt(result[k] / len(val_loader))
            print("MSE:{:.3f}".format(result[k]))
        elif k=='GAME':
            for i in range(4):
                result[k][i]/=len(val_loader)
                print("GAME{}:{:.3f}".format(i+1,result[k][i]))
        else:
            result[k]/=len(val_loader)
            print("{}:{:.3f}".format(k,result[k]))

    if not train and cfg['partition']:
        part_MAE_log_file_path=os.path.join(cfg['log_folder_path'],'log_metric_part_MAE_{}.txt'.format(time_index))
        part_MAE_log_file = open(part_MAE_log_file_path, 'a')
        print("\n [part MAE]")
        key_list=list(part_result.keys())
        key_list.sort()
        for k in key_list:
            print("{}:{}".format(k,np.mean(part_result[k])))
            part_MAE_log_file.writelines("{},{}\n".format(k,np.mean(part_result[k])))
        part_MAE_log_file.close()
    return result