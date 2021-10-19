# -*- coding: UTF-8 -*-
#@Time : 2021/7/15 16:41
#@File : predict.py
#@Software: PyCharm
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2 as cv
import torch
import numpy as np

from InputProcess import twoValue, clearNoise, saveImage
from Cut import smartSliceImg
from ReadLabel import ReadPic
from NeuralNet import AlexNet

def Predict():
    dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 10
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z',  # 35
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
    numinput, numoutput= 224*224,  36
    batch_size = 4
    net = AlexNet(numinput,numoutput,batch_size)
    PATH = "improved_model.pth"
    net.load_state_dict(torch.load(PATH))
    net.eval()

    ## 图像二值化处理
    outDir = 'static/processed/'
    path = "static/" + "images/" +  "test.jpg"
    image = Image.open(path)

    image = image.convert('L')
    twoValue(image, 120)
    clearNoise(image, 4, 1)
    # path1 = "test_processed/" + str(i) + ".jpg"
    # saveImage(path1, image.size)

    ## 图像分割
    # path = "test_processed/" + str(ii) + ".jpg"
    # img = Image.open(path)
    smartSliceImg(image, outDir, "test", count=4, p_w=2)

    ## 预测
    data_ = np.zeros((1, 4, 224, 224))
    data_ = data_.astype(np.float32)

    for j in range(4):
        path = outDir  + "test" + "_" + str(j) + ".png"
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (224, 224))
        # img = cv.medianBlur(img, 21)
        data_[0,  j, :, :] = img[0:224, 0:224]
    X = torch.from_numpy(data_)
    y_hat = net(X[0,0:4,:,:])
    pre = y_hat.argmax(dim=1).int()
    output = []
    for ii in range(4):
        a = dict[pre[ii]]
        output.append(a)
    return output