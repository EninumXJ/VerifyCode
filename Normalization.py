# -*- coding: UTF-8 -*-
#@Time : 2021/7/13 12:50
#@File : Mean&Std.py
#@Software: PyCharm
import numpy as np
import cv2
import os
from ReadLabel import ReadPic

def Normalize(img):
    imgs = img[:,:,:,:]
    normMean = 0.6951
    normStd = 0.4225
    mean = np.ones_like(imgs) * normMean
    imgs = (imgs/255 - mean)/normStd
    return imgs

if __name__=='__main__':
    img_path = "cut"
    label_path = "label.txt"
    imgs, label = ReadPic(img_path, label_path)
    imgs = Normalize(imgs)
    print(imgs[0,0])

# # img_h, img_w = 32, 32
# img_h, img_w = 20, 20  # 根据自己数据集适当调整，影响不大
# means, stdevs = 0, 0
# img_list = []
#
# imgs_path = 'cut/'
# imgs_path_list = os.listdir(imgs_path)
#
# len_ = len(imgs_path_list)
# i = 0
# for item in imgs_path_list:
#     img = cv2.imread(os.path.join(imgs_path, item),cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (img_w, img_h))
#     img = img[:, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     print(i, '/', len_)
#
# imgs = np.concatenate(img_list, axis=2)
# imgs = imgs.astype(np.float32) / 255.
#
#
# pixels = imgs.ravel()  # 拉成一行
# means = np.mean(pixels)
# stdevs = np.std(pixels)
#
# print("normMean = {}".format(means))
# print("normStd = {}".format(stdevs))
