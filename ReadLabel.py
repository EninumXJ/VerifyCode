# -*- coding: UTF-8 -*-
#@Time : 2021/7/8 10:07
#@File : ReadLabel.py
#@Software: PyCharm
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2 as cv

# 读取图片并将标签信息也读取进来
def ReadPic(img_path,label_path):
    dict=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', #10
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', #35
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    data_ = np.zeros((1,8000,224,224))
    data_ = data_.astype(np.float32)
    # data = np.zeros((224,224))
    # data = data.astype(np.float32)
    label= np.zeros((1,8000))
    label = label.astype(np.float32)
    for i in range(0,2000):
        for j in range(0,4):
            path = img_path + "/" + str(i) + "_"+ str(j) + ".png"
            img = cv.imread(path,cv.IMREAD_GRAYSCALE)
            img = cv.resize(img,(224,224))
            data_[0,4*i+j,:,:] = img[0:224,0:224]
    lines=[]
    for line in open(label_path,"r",encoding="utf-8"):
        lines.append(list(line))
    for i in range(0,2000):
        for ii in range(0,4):
            a = dict.index(lines[i][ii])
            if(a==2):
                label[0,i*4+ii] = 21
            elif(a>35):
                label[0,i*4+ii] = a-26
            else:
                label[0,i*4+ii] = a

    return data_, label

if __name__=='__main__':
    img_path = "cut"
    label_path = "label.txt"
    train, label = ReadPic(img_path,label_path)
    print(type(train))

