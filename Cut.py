# -*- coding: UTF-8 -*-
#@Time : 2021/7/8 15:05
#@File : Cut.py
#@Software: PyCharm
from PIL import Image

def smartSliceImg(img, outDir, ii,count=4, p_w=3):
    '''
    :param img:
    :param outDir:
    :param count: 图片中有多少个图片
    :param p_w: 对切割地方多少像素内进行判断
    :return:
    '''
    w, h = img.size
    pixdata = img.load()
    eachWidth = int(w / count)
    beforeX = 0
    for i in range(count):

        allBCount = []
        nextXOri = (i + 1) * eachWidth

        for x in range(nextXOri - p_w, nextXOri + p_w):
            if x >= w:
                x = w - 1
            if x < 0:
                x = 0
            b_count = 0
            for y in range(h):
                if pixdata[x, y] == 0:
                    b_count += 1
            allBCount.append({'x_pos': x, 'count': b_count})
        sort = sorted(allBCount, key=lambda e: e.get('count'))

        nextX = sort[0]['x_pos']
        box = (beforeX, 0, nextX, h)
        img.crop(box).save(outDir + str(ii) + "_" + str(i) + ".png")
        beforeX = nextX

if __name__=='__main__':
    for ii in  range(0, 2000):
        path = "result_processed/" + str(ii) + ".jpg"
        img = Image.open(path)
        outDir = 'cut/'
        smartSliceImg(img, outDir, ii,count=4, p_w=3)