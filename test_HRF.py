#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import argparse
import cv2
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_data import TrainData
from data.load_test_data import TestData
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from net.retinal_vasuclar_net import RetinalVasularSeg
from net.loss import dice_loss, cal_sen
from metrics import Evaluate
import config.config as cfg

def model_initial(model, model_name):
    # 加载预训练模型
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dictf = {k.replace('module.', ""): v for k, v in pretrained_dict.items() if k.replace('module.', "") in model_dict}
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dictf)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print("over")



def model_forward(model, test_data, patch_size, hh,ww, stride_y, stride_x):

    sy = math.ceil((hh - patch_size[0]) / stride_y) + 1
    sx = math.ceil((ww - patch_size[1]) / stride_x) + 1

    score_map = np.zeros((1, hh, ww)).astype(np.float32)
    cnt = np.zeros((1, hh, ww)).astype(np.int32)
    test_datas = []
    boxes = []
    for y in range(0, sy):
        ys = min(stride_y * y, hh - patch_size[0])
        for x in range(0, sx):
            xs = min(stride_x * x, ww - patch_size[1])

            test_patch = test_data[:, :, ys:ys + patch_size[0], xs:xs + patch_size[1]]
            test_datas.append(test_patch)
            boxes.append([ys, ys + patch_size[0], xs, xs + patch_size[1]])

    bsize = 16
    batch_nums = math.ceil(len(test_datas)/bsize)
    for i in range(batch_nums):
        test_patch = torch.cat(test_datas[i*bsize:(i+1)*bsize], dim=0)
        outputs_segb = model(test_patch)[0]
        outputs_softb = F.softmax(outputs_segb, dim=1)
        predb = torch.squeeze(outputs_softb[:, 1, :, :]).detach().cpu().numpy()

        for j in range(predb.shape[0]):
            y1, y2, x1, x2 = boxes[i*bsize +j]
            score_map[0, y1: y2, x1: x2] = score_map[0, y1: y2, x1: x2] + predb[j]

            cnt[0, y1: y2, x1: x2] = cnt[0, y1: y2, x1: x2] + 1

    # cv2.namedWindow("score_map", cv2.WINDOW_NORMAL)
    # cv2.imshow("score_map", score_map[0, :, :])
    #
    # cv2.waitKey(0)
    score_map = torch.tensor(score_map / cnt).cuda().float()

    return  score_map
from PIL import Image, ImageTk
def get_test_data(file_path, label_root):

    if "JPG" in file_path:
        label_path = label_root + os.path.split(file_path)[-1].replace(".JPG", ".tif")
        name = os.path.split(file_path)[-1].replace(".JPG", "0")
    else:
        label_path = label_root + os.path.split(file_path)[-1].replace(".jpg", ".tif")
        name = os.path.split(file_path)[-1].replace(".jpg", "0")

    img = Image.open(file_path)  #
    label = Image.open(label_path)  #
    height, width = img.size
    img = img.resize((height // 2, width // 2), resample=Image.BILINEAR)
    # label = label.resize((width // 2, width // 2), resample=Image.BILINEAR)

    data2D_ = np.array(img)
    data2D = np.concatenate([data2D_[:, :, 2:], data2D_[:, :, 1:2], data2D_[:, :, 0:1]], axis=2)
    label2D = np.array(label)

    data2D = (data2D - np.min(data2D)) / (np.max(data2D) - np.min(data2D))
    label2D = label2D / np.max(label2D)

    data2D = torch.tensor(data2D).permute(2, 0, 1)
    label2D = torch.tensor(label2D)


    return data2D, label2D, [width, height]





def test():


    from data.load_test_data import get_files
    file_path = "G:/vasular/HRFdatas/test/"
    label_path = "G:/vasular/HRFdatas/manualsegm/"
    file_list = []
    file_list1 = []
    file_list2 = []
    get_files(file_path, file_list1, "jpg")
    get_files(file_path, file_list2, "JPG")
    file_list = file_list1 + file_list2

    # Try to load models
    model = RetinalVasularSeg(in_channels=3,
                            out_channels=2,
                            img_size=[cfg.CROP_SIZE, cfg.CROP_SIZE],
                            feature_size=16,
                            num_heads=4,
                            depths=[3, 3, 3, 3],
                            dims=[32, 64, 128, 256],
                              hidden_size=256,
                            do_ds=True,
                            )

    model_name = "./outputs/teethseg_model_1000.pth"
    model_initial(model, model_name)

    model.cuda()
    model.eval()
    evaluate = Evaluate()

    tic = time.time()
    nums = 0
    sen_v, acc_v, spec_v =0, 0, 0
    patch_size = [cfg.CROP_SIZE, cfg.CROP_SIZE]
    stride_y, stride_x = 64, 64
    score_maps, test_labels =[], []

    for di in range(len(file_list)):
        file_path = file_list[di]
        print(file_path)
        test_data, test_label, sizem = get_test_data(file_path, label_path)

        test_data, test_label = torch.unsqueeze(test_data, dim=0), torch.unsqueeze(test_label, dim=0)

        with torch.no_grad():
            test_data = test_data.cuda().float()
            test_label = test_label.cuda()

            hh, ww = test_data.shape[-2:]
            nums = nums +1

            test_label[test_label >= 0.5] = 1
            score_map = model_forward(model, test_data, patch_size, hh, ww, stride_y, stride_x)
            #Upsampling the image will reduce the pixel value, so it is important to ensure that pixels with
            # a pixel value greater than 0.5 remain the target as much as possible after magnification.
            # score_map[score_map>=0.5] =1
            # score_map = F.interpolate(torch.unsqueeze(score_map,dim=0), (sizem[0], sizem[1]), mode='bilinear')
            # score_map = torch.squeeze(score_map,dim=0)

            #cal Acc
            sen_v_, acc_v_, spec_v_ = cal_sen(score_map, test_label.long())

            # pred = torch.squeeze(outputs_soft[:, 1, :, :]).cpu().numpy() *255
            # img = torch.squeeze(test_data).permute(1, 2, 0).cpu().numpy()*255

            # cv2.imwrite("./outputs/"+ str(nums) + "_"+str(acc_v_.item()) + "p.png", pred)
            # cv2.imwrite("./outputs/" + str(nums) + "_" + str(acc_v_.item()) + "m.png", img)
            print(nums)
            sen_v = sen_v + sen_v_.item()
            acc_v = acc_v + acc_v_.item()
            spec_v = spec_v + spec_v_.item()
            score_maps.append(torch.squeeze(score_map).cpu().numpy())
            test_labels.append(torch.squeeze(test_label.long()).cpu().numpy())
    score_maps = np.array(score_maps)
    test_labels = np.array(test_labels)
    evaluate.add_batch(test_labels, score_maps)
    result = evaluate.save_all_result()
    print(result)
    print("sen_v = ", sen_v/nums, "  ", "acc_v = ", acc_v/nums, "  ", "spec_v = ", spec_v/nums, "  ")



if __name__ == "__main__":

   test()

