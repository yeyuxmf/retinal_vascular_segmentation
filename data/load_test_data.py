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


class TestData():
    def __init__(self, file_root, label_path, train_flag = False):
        self.data_dir = file_root
        self.label_path = label_path
        self.train_flag = train_flag

        self.train_list = None
        self.prepare(self.data_dir)
        # self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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

        label_path = self.label_path + os.path.split(file_path)[-1].replace("_test.tif", "_manual1.gif")
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


        data2D = (data2D - np.min(data2D)) / (np.max(data2D) - np.min(data2D))
        label2D = label2D / np.max(label2D)


        data2D = torch.tensor(data2D).permute(2, 0, 1)
        label2D = torch.tensor(label2D)


        # cv2.namedWindow("data2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("data2D", data2D[0, :, :].numpy())
        # cv2.namedWindow("label2D", cv2.WINDOW_NORMAL)
        # cv2.imshow("label2D", label2D)
        #
        # cv2.waitKey(0)



        return data2D, label2D



