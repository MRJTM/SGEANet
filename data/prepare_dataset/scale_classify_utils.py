"""
@File       : scale_classify_utils.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/31
@Desc       : functions of generating scale classification gt
"""
import cv2
import numpy as np
import copy
import progressbar
import scipy
import scipy.ndimage
from scipy import spatial
from utils.my_utils import create_progressbar

def generate_scale_classify_map(image,gt_points,bboxs,K,threshold_list):
    point_gts = gt_points
    """build tree and query"""
    # print("pts:",pts)
    leafsize = 2048

    # get det bbox center list
    det_center_list = []
    det_size_list = []
    for box in bboxs:
        det_center_list.append([box[4], box[5]])
        det_size_list.append(box[6])

    # build kdtree of det, 找到每个gt的点最近的几个detection
    det_tree = scipy.spatial.KDTree(det_center_list.copy(), leafsize=leafsize)
    det_distances, det_locations = det_tree.query(point_gts, k=4)

    # build kdtree of gt point，找到每个gt的点最近的几个gt point
    point_tree = scipy.spatial.KDTree(point_gts.copy(), leafsize=leafsize)
    point_distances, point_locations = point_tree.query(point_gts, k=4)

    """generate scale class map"""
    class_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(len(point_gts)):
        # get gt point
        point = point_gts[i]

        # 每个gt的点处画一个圆上去
        if int(round(point[1])) < image.shape[0] and int(round(point[0])) < image.shape[1]:
            # compute average size of 4NN det boxs
            total_size = 0
            num_det_locations = min(len(det_locations[i]), len(det_center_list))
            # print("num_det_locations[{}]={}".format(i,num_det_locations))
            for j in range(num_det_locations):
                total_size += det_size_list[det_locations[i][j]]

            det_size = total_size / num_det_locations

            # compute of average size of 4NN geometry-adaptive size
            num_point_locations = len(point_distances[i])
            # print("num_point_locations:",num_point_locations)
            ga_size = 0
            for j in range(num_point_locations - 1):
                ga_size += point_distances[i][j + 1]
            if num_point_locations - 1 > 0:
                ga_size /= (num_point_locations - 1)
            if ga_size < 0.00001:
                ga_size = 15

            # comapre det_size and ga_size to get final size
            head_size=min(det_size,ga_size)*0.6
#             print('head_size:',head_size)

            """产生圆圈图"""
            # 根据scale确定颜色,人头越小，颜色越亮
            color=0
            for k in range(K - 1):
                if head_size < threshold_list[k] and color == 0:
                    color=K-k-1

            # 确定圆心，半径
            center=(int(round(point[0])),int(round(point[1])))
            radius=int(round(head_size))
            # 画圆
            class_map=cv2.circle(class_map,center,radius,color,thickness=-1)

    return class_map