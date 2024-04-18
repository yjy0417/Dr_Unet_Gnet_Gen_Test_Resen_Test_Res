# -*- coding: UTF-8 -*-
# Created on 2024/4/16-15:52
import glob
from pathlib import Path

from imageio import imsave, imread
import numpy as np, os, glob
import skimage.transform as trans
from model import *


def testGenerator(test_path, target_size=(128, 128), flag_multi_class=False, as_gray=True):
    image_name_arr = glob.glob(os.path.join(test_path, "*.png"))
    cnt = 0
    for index, item in enumerate(image_name_arr):
        img = imread(item)
        img = img / 255
        if img.shape[0] != target_size[0] | img.shape[1] != target_size[1]:
            img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, img.shape)
        yield img
        cnt += 1
        if cnt > 10:
            yield None


def saveResult(test_path, save_path, npyfile, flag_multi_class=False, num_class=2, start_idx=0, batch_size=32):
    image_name_arr = glob.glob(os.path.join(test_path, "*.png"))
    string = image_name_arr[0]
    word = 'image'
    index = string.find(word) + 6
    for i in range(min(batch_size, len(npyfile))):
        item = image_name_arr[start_idx + i]
        print(item)
        img = npyfile[i, :, :, 0]
        imsave(Path(save_path, item[index:]),
               np.uint8(img * 255))  # index: to take only the image name that was read before


test_path = r'D:\Pycharm_Projects\UNet\DataV1\CV4\test\crops\image'
windowLen = 128


