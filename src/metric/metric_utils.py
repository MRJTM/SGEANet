"""
@File       : metric_utils.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/22
@Desc       :
"""

import numpy as np

# 计算GAME误差
def compute_GAME(pred, gt, K=4):
    result_list = []
    # 获取density map的尺寸
    h = pred.shape[0]
    w = pred.shape[1]
    for k in range(K):
        # 计算划分区域的gap
        gap_h = h // (k + 1)
        gap_w = w // (k + 1)
        # 一张一张来统计loss，最后加起来
        game = 0
        # 把每块的MSE加起来
        for i in range(k + 1):
            for j in range(k + 1):
                x1 = j * gap_w
                y1 = i * gap_h
                if i < k:
                    y2 = y1 + gap_h
                else:
                    y2 = h
                if j < k:
                    x2 = x1 + gap_w
                else:
                    x2 = w
                pred_block = pred[y1:y2, x1:x2]
                gt_block = gt[y1:y2, x1:x2]
                pred_num = np.sum(pred_block)
                gt_num = np.sum(gt_block)
                # print("h={},w={},k={},i={},j={},x1={},y1={},x2={},y2={},pred={},gt={}".format(h, w, k, i, j, x1, y1, x2, y2,pred_num,gt_num))
                game += abs(pred_num - gt_num)
        result_list.append(game)

    return result_list