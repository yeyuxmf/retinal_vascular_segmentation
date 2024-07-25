#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_data import TrainData, data_crop
from data.load_test_data import TestData
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from net.retinal_vasuclar_net import RetinalVasularSeg
from net.unetrpp_ff import RetinalVasularSegFF
from net.UNetFamily import U_Net, AttU_Net, Dense_Unet
# from net.Hessian import HessianNet
from net.loss import dice_loss, FocalLoss2d, WeightedFocalLoss, cal_sen
from test import model_forward
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


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('./outputs/' + args.exp_name):
        os.makedirs('./outputs/' + args.exp_name)
    if not os.path.exists('./outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('./outputs/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls.py outputs' + '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, io):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "E:/vasular/DRIVE/training/images/"
    label_path = "E:/vasular/DRIVE/training/1st_manual/"
    # file_path = "G:/vasular/CHASEDB1/images/"
    # label_path = "G:/vasular/CHASEDB1/1st_label/"
    # file_path = "G:/vasular/STAREdatabase/images/"
    # label_path = "G:/vasular/STAREdatabase/labels-ah/"
    # file_path = "G:/vasular/HRFdatas/images/"
    # label_path = "G:/vasular/HRFdatas/images/"
    train_loader = DataLoader(TrainData(file_path, label_path, train_flag = True), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    file_path = "E:/vasular/DRIVE/test/images/"
    label_path = "E:/vasular/DRIVE/test/1st_manual/"
    # file_path = "G:/vasular/CHASEDB1/test/"
    # label_path = "G:/vasular/CHASEDB1/1st_label/"
    # file_path = "G:/vasular/STAREdatabase/test/"
    # label_path = "G:/vasular/STAREdatabase/labels-ah/"
    # file_path = "G:/vasular/HRFdatas/test/"
    # label_path = "G:/vasular/HRFdatas/test/"
    test_loader = DataLoader(TestData(file_path, label_path, train_flag = False), num_workers=0,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = RetinalVasularSegFF(in_channels=3,
                            out_channels=2,
                            img_size=[cfg.CROP_SIZE, cfg.CROP_SIZE],
                            feature_size=16,
                            num_heads=4,
                            depths=[3, 3, 3, 3],
                            dims=[32, 64, 128, 256],
                              hidden_size=256,
                            do_ds=True,
                            )
    # model = HessianNet(in_channel=3, out_channels=2)
    # model = U_Net(img_ch=3, output_ch=2, fea_channels=64)
    model_name = "./outputs/teethseg_model_200.pth"
    # model_initial(model, model_name)


    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.use_sgd:
        print("Use SGD")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)

    decay_lrs = cfg.DECAY_STEPS
    focalLoss = FocalLoss2d(gamma=2)#WeightedFocalLoss()#

    model.cuda()
    model.train()
    scaler = GradScaler()
    inter_nums = len(train_loader)
    total_acc = 0

    for epoch in range(0, args.epochs):
        ####################
        # Train
        ####################
        if epoch in decay_lrs:
            lr = decay_lrs[epoch]
            adjust_learning_rate(opt, lr)
            print('adjust learning rate to {}'.format(lr))

        train_loss = 0.0
        loss_dice = 0
        sen_v = 0
        acc_v = 0
        spec_v = 0
        # for data, edges, label in train_loader:
        tic = time.time()
        nums = 0
        model.train()
        for train_data, train_label, weight_ in train_loader:

            train_data, train_label, train_weight_ = data_crop(train_data, train_label, weight_)

            train_data = train_data.cuda().float()
            train_label = train_label.cuda().float()
            train_weight_ = train_weight_.cuda().float()

            nums = nums +1
            opt.zero_grad()
            with autocast():
                outputs_seg, out2, _ = model(train_data)

                train_label[train_label >= 0.5] = 1
                loss_seg = F.cross_entropy(outputs_seg, train_label.long())
                # loss_seg = F.binary_cross_entropy_with_logits(outputs_seg[:,0, :, :], train_label)
                sec_loss = focalLoss(outputs_seg, train_label[:, :, :].long())

                outputs_soft = torch.softmax(outputs_seg, dim=1)[:, 1, :, :]#F.sigmoid(outputs_seg[:, 0, :, :])#
                loss_dice_ = dice_loss(outputs_soft, train_label[:, :, :])

                sen_v_, acc_v_, spec_v_ = cal_sen(outputs_soft, train_label[:, :, :].long())

                loss = loss_seg +loss_dice_+ 2*sec_loss  #

            scaler.scale(loss).backward()
            # Unscales gradients and calls
            # or skips optimizer.step()
            scaler.step(opt)
            # Updates the scale for next iteration
            scaler.update()

            train_loss += loss.item()
            loss_dice += loss_dice_.item()
            sen_v += sen_v_.item()
            acc_v += acc_v_.item()
            spec_v += spec_v_.item()
            if nums % cfg.VIEW_NUMS == 0:
                toc = time.time()
                train_loss = train_loss/ (cfg.VIEW_NUMS)
                loss_dice = loss_dice / (cfg.VIEW_NUMS)
                sen_v = sen_v/(cfg.VIEW_NUMS)
                acc_v = acc_v/(cfg.VIEW_NUMS)
                spec_v = spec_v/(cfg.VIEW_NUMS)

                print("lr = ", opt.param_groups[0]['lr'])
                outstr = 'epoch %d /%d,epoch %d /%d, loss: %.6f, loss_dice: %.6f, sen_v: %.6f, acc_v: %.6f, spec_v: %.6f, const time: %.6f' % (
                 epoch,args.epochs, nums, inter_nums, train_loss, loss_dice, sen_v, acc_v, spec_v, toc - tic)

                io.cprint(outstr)
                train_loss = 0.0
                loss_dice = 0
                sen_v = 0
                acc_v = 0
                spec_v = 0
                tic = time.time()
        if 0 == epoch % 10 and epoch>300:
            test_nums = 0
            loss_dice, sen_v, acc_v, spec_v =0, 0, 0, 0
            model.eval()
            patch_size = [cfg.CROP_SIZE, cfg.CROP_SIZE]
            stride_y, stride_x = 128, 128
            for test_data, test_label in test_loader:
                test_data = test_data.cuda().float()
                test_label = test_label.cuda().float()
                nums = nums + 1
                test_nums = test_nums + 1
                hh, ww = test_label.shape[-2:]
                with autocast():
                    score_map = model_forward(model, test_data, patch_size, hh, ww, stride_y, stride_x)
                    train_label[train_label > 0.5] = 1
                    loss_dice_ = dice_loss(score_map, test_label.long())
                    sen_v_, acc_v_, spec_v_ = cal_sen(score_map.detach(), test_label.detach().long())
                loss = 1.0 * (loss_dice_).item()
                sen_v = sen_v + sen_v_.item()
                acc_v = acc_v + acc_v_.item()
                spec_v = spec_v + spec_v_.item()
                loss_dice = loss_dice + loss_dice_.item()
            sen_v = sen_v/test_nums
            acc_v = acc_v/test_nums
            spec_v = spec_v/test_nums
            loss_dice = loss_dice/test_nums
            toc = time.time()
            outstr = 'epoch %d, loss: %.6f, loss_dice: %.6f, sen_v: %.6f, acc_v: %.6f, spec_v: %.6f, const time: %.6f' % (
                epoch, loss, loss_dice, sen_v, acc_v, spec_v, toc - tic)
            io.cprint("test   "+outstr)
            if (total_acc < (sen_v*0.35+acc_v*0.35+spec_v*0.35)):
                total_acc = (sen_v * 0.35 + acc_v * 0.35 + spec_v * 0.35)
                torch.save({'model': model.state_dict(), 'epoch': epoch},
                           'outputs/' + str(sen_v)+"_"+ str(acc_v)+ "_"+str(spec_v) + '.pth')
        if (epoch) % cfg.SAVE_MODEL == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/teethseg_model_' + str(epoch)+ '.pth')


if __name__ == "__main__":

    AI = torch.rand(7, 7)
    mask = torch.tensor([[True, True,True,True, False, False, False]])

    attn = AI.masked_fill(mask == 0, 0)
    print(AI)
    print(" ")
    print(attn)


    torch.backends.cudnn.enabled = True
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='cls_1024', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1001, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=2*1e-3, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)

