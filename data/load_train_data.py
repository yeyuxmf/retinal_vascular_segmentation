import os
import copy
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
import PIL
from torchvision import transforms
import config.config as cfg

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(file_dir, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)

def AddGaussNoise(src,sigma):
    mean = 0
    # 获取图片的高度和宽度
    height, width, channels = src.shape[0:3]
    gauss = np.random.normal(mean,sigma,(height,width,channels))


    noisy_img = src + gauss
    noisy_img = np.clip(noisy_img, 0, 1)


    # cv2.namedWindow("data2D", cv2.WINDOW_NORMAL)
    # cv2.imshow("data2D", noisy_img)
    #
    # cv2.waitKey(0)
    return noisy_img

class TrainData():
    def __init__(self, file_root, label_path, train_flag = False):
        self.data_dir = file_root
        self.label_path = label_path
        self.train_flag = train_flag

        self.train_list = None
        self.prepare(self.data_dir)


        self.transformsRotate  = transforms.RandomRotation(degrees=[-45, 45], interpolation=PIL.Image.BILINEAR)
        self.transformGray = transforms.RandomGrayscale(p=0.5)

        self.transformVFlip = transforms.RandomVerticalFlip(p=0.5)
        self.transformHFlip = transforms.RandomHorizontalFlip(p=0.5)
        # self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # cl1 = clahe.apply(img)

    def prepare(self, file_path):

        file_list = []
        get_files(file_path, file_list, "tif") #drive
        # get_files(file_path, file_list, "jpg") #CHASEDB1
        # get_files(file_path, file_list, "ppm") #STAREdatabase
        # get_files(file_path, file_list, "im.npy")#HRFdatas

        self.train_list = file_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):

        file_path = self.train_list[item]

        label_path = self.label_path + os.path.split(file_path)[-1].replace("_training.tif", "_manual1.gif")
        # label_path = self.label_path + os.path.split(file_path)[-1].replace(".jpg", "_1stHO.png")
        #label_path = self.label_path + os.path.split(file_path)[-1].replace(".ppm", ".ah.ppm")
        # label_path = self.label_path + os.path.split(file_path)[-1].replace("im.npy", "label.npy")


        img = cv2.imread(file_path)
        label = Image.open(label_path)#cv2.imread(label_path, 0)#
        label = np.array(label)
        # img = np.load(file_path)
        # label = np.load(label_path)

        data2D = img
        label2D = label

        data2D = (data2D - np.min(data2D))/(np.max(data2D) - np.min(data2D))
        label2D = label2D / np.max(label2D)


        data2D = torch.tensor(data2D).permute(2, 0, 1)
        label2D = torch.tensor(label2D)

        label2D = torch.unsqueeze(label2D, dim=0)
        mdata = torch.cat([label2D, data2D], dim=0)


        mdata = self.transformVFlip(mdata)
        mdata = self.transformHFlip(mdata)
        #random roatte
        flag = np.random.randint(0, 2, 1)[0]
        # if flag==1:
        mdata = self.transformsRotate(mdata)

        label2D = mdata[0,:,:]#.numpy()
        data2D = mdata[1:,:,:]#.permute(1, 2, 0).numpy()



        scale = np.random.randint(8, 14, 1)[0] / 10
        #
        data2D = torch.pow(data2D, scale)

        weight_ = cv2.GaussianBlur(label2D.numpy(), (5, 5), 1)

        #to Gray
        # data2D = self.transformGray(data2D)
        # data2D_ = data2D.permute(1, 2, 0).numpy()
        # label2D_ = label2D.numpy()
        # cv2.namedWindow("data2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("data2D", data2D_)
        # cv2.namedWindow("label2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("label2D", weight_)
        #
        # cv2.waitKey(0)



        return data2D, label2D, weight_


def data_crop(train_data, train_label, weight_):

    bn, c, height, width = train_data.shape

    rangh = height - cfg.CROP_SIZE-1
    rangw = width - cfg.CROP_SIZE-1


    nums = 4
    train_data_ = torch.zeros((nums*bn, 3,cfg.CROP_SIZE, cfg.CROP_SIZE))
    train_label_ = torch.zeros((nums*bn,cfg.CROP_SIZE, cfg.CROP_SIZE))
    train_weight_ = torch.zeros((nums*bn,cfg.CROP_SIZE, cfg.CROP_SIZE))


    for b in range(bn):
        for i in range(nums):
            offh = np.random.randint(0, rangh, 1)[0]
            offw = np.random.randint(0, rangw, 1)[0]
            # print(height, " ", width, " ",  offh+cfg.CROP_SIZE, "   ", offw+cfg.CROP_SIZE)
            train_data_[nums*b+i,:, :, :] = train_data[b, :, offh:offh+cfg.CROP_SIZE, offw:offw+cfg.CROP_SIZE]
            train_label_[nums*b+i, :, :] = train_label[b, offh:offh+cfg.CROP_SIZE, offw:offw+cfg.CROP_SIZE]
            train_weight_[nums*b+i, :, :] = weight_[b, offh:offh+cfg.CROP_SIZE, offw:offw+cfg.CROP_SIZE]
            # data2D = train_data[0, :, offh:offh+cfg.CROP_SIZE, offw:offw+cfg.CROP_SIZE].permute(1, 2, 0).numpy()
            # label2D = train_label[0, offh:offh+cfg.CROP_SIZE, offw:offw+cfg.CROP_SIZE].numpy()
            # cv2.namedWindow("data2D", cv2.WINDOW_NORMAL)
            # cv2.imshow("data2D", data2D)
            # cv2.namedWindow("label2D", cv2.WINDOW_NORMAL)
            # cv2.imshow("label2D", label2D)
            #
        # cv2.waitKey(0)

    return train_data_, train_label_, train_weight_









