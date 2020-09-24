"""
@File       : density_map_utils.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/31
@Desc       : contains functions of generating density map
"""

import cv2
import numpy as np
import copy
import progressbar
import scipy
import scipy.ndimage
from scipy import spatial
from src.utils.common_utils import create_progressbar

# 制作gaussian kernel
def generate_gaussian_kernel(kernel_size=(15, 15), sigma=3):
    rows=kernel_size[0]
    cols=kernel_size[1]
    if rows%2==1:
        m1=int((rows-1)/2)
        m2=int((rows-1)/2)+1
    else:
        m1=int(rows/2)
        m2=int(rows/2)

    if cols%2==1:
        n1 = int((cols - 1) / 2)
        n2 = int((cols - 1) / 2) + 1
    else:
        n1=int(cols/2)
        n2=int(cols/2)

    y, x = np.ogrid[-m1:m2, -n1:n2]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# 每个ground truth的点处，填充一个正方形的gaussian kernel
def gaussian_disperse(density_map, kernel_size=15, sigma=3):
    max_rows = density_map.shape[0]
    max_cols = density_map.shape[1]
    density_map_blur = np.zeros((max_rows, max_cols))

    # 根据density_map中的点的位置，放上高斯核
    rows = np.where(density_map == 1)[0]
    cols = np.where(density_map == 1)[1]
    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        # print('position:({},{})'.format(row,col))
        row1 = row - int(np.floor(kernel_size / 2))
        row2 = row + int(np.floor(kernel_size / 2))
        col1 = col - int(np.floor(kernel_size / 2))
        col2 = col + int(np.floor(kernel_size / 2))
        # print("r1:{},r2:{},col1:{},col2:{}".format(row1,row2,col1,col2))
        change_kernel = False
        if row1 < 0:
            row1 = 0
            change_kernel = True
        if row2 > max_rows - 1:
            row2 = max_rows - 1
            change_kernel = True
        if col1 < 0:
            col1 = 0
            change_kernel = True
        if col2 > max_cols - 1:
            col2 = max_cols - 1
            change_kernel = True

        if change_kernel == False:
            size = (kernel_size, kernel_size)
        else:
            size = (row2 - row1 + 1, col2 - col1 + 1)

        gaussian_kernel = generate_gaussian_kernel(kernel_size=size, sigma=sigma)
        density_map_blur[row1:row2 + 1, col1:col2 + 1] += gaussian_kernel

    return density_map_blur

# input 2-D point gt map, output gt density map
def generate_ga_density_map(gt,max_size=48,dis_sigma_rate=8):
    # print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    # get point annotation list
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    widgets = ["density generation process : ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(pts),
                                   widgets=widgets).start()
    # put gaussian kernel to each point
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            head_size=(distances[i][1]+distances[i][2]+distances[i][3])/3
            head_size=min(head_size,max_size)
            sigma = head_size/dis_sigma_rate
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        pbar.update(i)
    pbar.finish()
    print ('done.\n')
    return density

# 输入gt points, 以及detections，然后输出一张集合det和ga的density map
def generate_det_ga_density_map(image,gt_points,bboxs):
    point_gts=gt_points
    """build tree and query"""
    # print("pts:",pts)
    leafsize = 2048

    # get det bbox center list
    det_center_list = []
    det_size_list = []
    for box in bboxs:
        det_center_list.append([box[4], box[5]])
        det_size_list.append(box[6])
    # print("size(det_center_list):",len(det_center_list))
    # print("size(det_size_list):", len(det_size_list))

    # build kdtree of det
    det_tree = scipy.spatial.KDTree(det_center_list.copy(), leafsize=leafsize)
    det_distances, det_locations = det_tree.query(point_gts, k=4)

    # build kdtree of gt point
    point_tree = scipy.spatial.KDTree(point_gts.copy(), leafsize=leafsize)
    point_distances, point_locations = point_tree.query(point_gts, k=4)

    """generate density map"""
    print("generating det ga density map")
    SAANet_density_gt = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    pbar = create_progressbar("process", len(point_gts))
    for i in range(len(point_gts)):
        # get gt point
        point = point_gts[i]

        # point within the image would be put a gaussian kernel
        if int(round(point[1]))<image.shape[0] and int(round(point[0]))<image.shape[1]:
            # compute average size of 4NN det boxs
            total_size = 0
            num_det_locations = min(len(det_locations[i]),len(det_center_list))
            # print("num_det_locations[{}]={}".format(i,num_det_locations))
            for j in range(num_det_locations):
                total_size += det_size_list[det_locations[i][j]]

            det_size = total_size / num_det_locations

            # compute of average size of 4NN geometry-adaptive size
            num_point_locations=len(point_distances[i])
            # print("num_point_locations:",num_point_locations)
            ga_size=0
            for j in range(num_point_locations-1):
                ga_size += point_distances[i][j+1]
            if num_point_locations-1>0:
                ga_size/=(num_point_locations-1)
            if ga_size<0.00001:
                ga_size=15

            # comapre det_size and ga_size to get final size
            if det_size < ga_size:
                sigma_SAA = det_size * 3 * 0.1
            else:
                sigma_SAA = ga_size * 3 * 0.1

            # print("sigma_SAA=",sigma_SAA)

            # generate density map
            pt2d = np.zeros(SAANet_density_gt.shape, dtype=np.float32)
            pt2d[int(round(point[1])), int(round(point[0]))] = 1.

            SAANet_gaussian_kernel = scipy.ndimage.filters.gaussian_filter(pt2d, sigma_SAA, mode='constant')
            SAANet_gaussian_kernel *= 1 / np.sum(SAANet_gaussian_kernel)
            SAANet_density_gt += SAANet_gaussian_kernel

        pbar.update(i)
    pbar.finish()

    return  SAANet_density_gt