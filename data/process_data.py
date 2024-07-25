import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from data.load_test_data import get_files

def dataPreprocess():

    file_path = "G:/vasular/HRFdatas/test/"
    label_root  = "G:/vasular/HRFdatas/manualsegm/"

    file_list = []
    get_files(file_path, file_list, "JPG")


    for i in range(len(file_list)):
        data_path = file_list[i]
        label_path = label_root + os.path.split(data_path)[-1].replace(".JPG", ".tif")
        name = os.path.split(data_path)[-1].replace(".JPG", "")

        img = Image.open(data_path)#
        label = Image.open(label_path)#
        height, width= img.size
        img = img.resize((height//2, width//2),resample=Image.BILINEAR)
        label = label.resize((height//2, width//2),resample=Image.BILINEAR)

        data2D_ = np.array(img)
        data2D_ = np.concatenate([data2D_[:, :, 2:], data2D_[:, :, 1:2], data2D_[:, :, 0:1]], axis =2)
        label2D_ = np.array(label)

        np.save(file_path+name+"im.npy", data2D_)
        np.save(file_path+name+"label.npy", label2D_)


        # cv2.namedWindow("data2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("data2D", data2D_)
        # cv2.namedWindow("label2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("label2D", label2D_)
        #
        # cv2.waitKey(0)


import cv2
import numpy as np
import math


def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


# 根据半径计算权重参数矩阵
g_para = {}


def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


# 常规的ACE实现
def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh = []
    zw = []
    n = 0
    while n < radius:
        zh.append(0)
        zw.append(0)
        n += 1
    for n in range(height):
        zh.append(n)
    for n in range(width):
        zw.append(n)
    n = 0
    while n < radius:
        zh.append(height - 1)
        zw.append(width - 1)
        n += 1
    # print(zh)
    # print(zw)

    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


# 单通道ACE快速增强实现
def zmIceFast(I, ratio, radius):
    # print(I)
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) / 2), int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


# rgb三通道分别增强 ratio是对比度增强因子 radius是卷积模板半径
def zmIceColor(I, ratio=4, radius=3):
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res


def ACE_MOTHOD():


    file_path = "G:/vasular/DRIVE/test/images/"
    label_root = "G:/vasular/DRIVE/test/1st_manual/"

    file_list = []
    get_files(file_path, file_list, "tif")


    for i in range(len(file_list)):
        data_path = file_list[i]
        label_path = label_root + os.path.split(data_path)[-1].replace("_test.tif", "_manual1.gif")
        name = os.path.split(data_path)[-1].replace(".tif", "")

        img = cv2.imread(data_path)
        label = Image.open(label_path)#cv2.imread(label_path, 0)#
        label = np.array(label)

        data2D_  = (img / 255.0).astype(np.float32)
        data2D_ = zmIceColor(data2D_)*255

        np.save(file_path+name+"im.npy", data2D_)
        np.save(file_path+name+"label.npy", label)


        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img[:, :,1])
        # cv2.namedWindow("data2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("data2D", data2D_)
        # cv2.namedWindow("label2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("label2D", label)
        #
        # cv2.waitKey(0)


if __name__ =="__main__":

    #dataPreprocess()

    ACE_MOTHOD()