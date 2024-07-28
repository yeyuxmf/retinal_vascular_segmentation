
from __future__ import print_function
import os
import time
import argparse
import cv2
import math
import torch
import numpy as np
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

    bsize = 8
    batch_nums = math.ceil(len(test_datas)/bsize)

    for i in range(batch_nums):
        krangv = (i+1)*bsize if (i+1)*bsize < len(test_datas) else len(test_datas)
        test_patch = torch.cat(test_datas[i*bsize: krangv], dim=0)
        outputs_segb = model(test_patch)[0]
        # outputs_softb = F.sigmoid(outputs_segb)
        outputs_softb = torch.softmax(outputs_segb, dim=1)[:, 1, :, :]
        predb = torch.squeeze(outputs_softb).detach().cpu().numpy()

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