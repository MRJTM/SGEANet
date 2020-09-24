"""
@File       : synthetic_utils.py
@Author     : Cao Zhijie
@Email      : cao_zhijie@sjtu.edu.cn
@Date       : 2020/5/31
@Desc       : functions of generating synthetic image
"""

import cv2
import numpy as np
import scipy
import scipy.ndimage
from scipy import spatial
from src.utils.common_utils import create_progressbar

# input 2-D point gt map, output a ga stnthetic image
def generate_ga_syn_image(gt,max_size=48):
    H=gt.shape[0]
    W=gt.shape[1]
    syn_img = np.ones((H,W,3), dtype=np.uint8)*255
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return syn_img

    """use kdtree to get scale of each point"""
    # get point annotation list
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    """draw a circle for each point"""
    pbar=create_progressbar(progress="generating syn_img",max_len=len(pts))
    # put gaussian kernel to each point
    for i, pt in enumerate(pts):
        if gt_count > 1:
            head_size=(distances[i][1]+distances[i][2]+distances[i][3])/3
            head_size=min(head_size,max_size)
        else:
            head_size=max_size

        head_size=head_size*0.8

        # draw circles
        syn_img=cv2.circle(syn_img,center=(pt[0],pt[1]),radius=int(head_size//2),
                           color=(0,0,0),thickness=-1)
        pbar.update(i)
    pbar.finish()
    print ('done.\n')
    return syn_img

