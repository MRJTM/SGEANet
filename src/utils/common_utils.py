import progressbar
import numpy as np
import cv2
from progressbar import *

# 创建一个进度条
def create_progressbar(progress="",max_len=100):
    widgets = ['{}: '.format(progress), Percentage(), ' ', Bar('#'), ' ',
               Timer(),' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=max_len).start()

    return pbar

def fix_str_len(s,fix_len=20):
    new_s=s
    if len(s)<fix_len:
        new_s+=' '*(fix_len-len(s))

    return new_s


