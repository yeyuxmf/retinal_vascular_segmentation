import cv2
import numpy as np
import torch
from torch.nn import functional as F
import config.config as cfg

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def secondary_losses(pred, train_label):
    train_label = torch.unsqueeze(train_label, dim=1)
    depth, height = cfg.IMG_SIZE//3, cfg.IMG_SIZE//3
    train_label = F.interpolate(train_label, (depth, height), mode='nearest')
    train_label = torch.squeeze(train_label, dim=1)


    loss_seg = F.cross_entropy(pred, train_label.long())
    outputs_soft = F.softmax(pred, dim=1)
    loss_dice_ = dice_loss(outputs_soft[:, 1, :, :], train_label[:, :, :])
    loss = 1.0 * (loss_dice_ + loss_seg)
    return  loss



class FocalLoss2d(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average # 对batch里面的数据取均值/求和

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').view(-1)#

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()



def cal_sen(outputs_soft, train_label):

    bn, h, w = outputs_soft.shape
    Sen = 0
    Acc = 0
    Spec = 0
    nm = 0
    for i in range(bn):
        b_data = torch.gt(outputs_soft[i], 0.5).float()#[ph: ph+height, pw:pw+width], 0.5).float()
        b_label = train_label[i]#[ph: ph+height, pw:pw+width].float()
        TP = torch.sum(b_data*b_label)
        FN = torch.sum((1 - b_data) *b_label)

        TN = torch.sum((1 - b_data) * (1- b_label))
        FP = torch.sum((b_data) * (1- b_label))

        if (TP + FN) ==0:
            Sen = Sen + 0
            nm = nm +1
        else:
            Sen = Sen + TP / (TP + FN)


        Acc = Acc + (TP + TN)/(TP + FN + TN + FP)
        Spec = Spec + (TN)/(TN + FP)

    Sen = Sen / (bn-nm)
    Acc = Acc / bn
    Spec = Spec / bn

    return Sen, Acc, Spec









