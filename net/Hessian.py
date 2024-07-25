import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



class HessianLayerD(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, bias=False, sigma=5):
        super().__init__()
        sigma = 3
        pads = ksize//2
        self.sigmaK = sigma
        self.sigma = sigma/3
        self.H_arange = torch.arange(-self.sigmaK, self.sigmaK+1)
        self.W_arange = torch.arange(-self.sigmaK, self.sigmaK+1)
        self.X, self.Y = torch.meshgrid(self.W_arange, self.H_arange, indexing='ij')
        self.X2 = torch.pow(self.X, 2)
        self.Y2 = torch.pow(self.Y, 2)
        self.DGaxx = 1/ (2 * torch.pi*self.sigma**4) * (self.X2 / (self.sigma**2) -1) * torch.exp(-(self.X2 + self.Y2) / (2 * (self.sigma**2)))
        self.DGaxy = 1/ (2 * torch.pi*self.sigma**6) * (self.X * self.Y) * torch.exp(-(self.X2 + self.Y2) / (2 * (self.sigma**2)))
        self.DGayy = torch.t(self.DGaxx)

        self.ksize = self.sigmaK*2+1
        self.weightxx = nn.Parameter(data=self.DGaxx, requires_grad=False).view(1, 1, self.ksize, self.ksize).expand(in_channels, in_channels, self.ksize, self.ksize).cuda().float()
        self.weightxy = nn.Parameter(data=self.DGaxy, requires_grad=False).view(1, 1, self.ksize, self.ksize).expand(in_channels, in_channels, self.ksize, self.ksize).cuda().float()
        self.weightyy = nn.Parameter(data=self.DGayy, requires_grad=False).view(1, 1, self.ksize, self.ksize).expand(in_channels, in_channels, self.ksize, self.ksize).cuda().float()

        self.param = nn.Parameter(torch.ones(1, out_channels, 1, 1),requires_grad=True)  # fixed sin-cos embedding

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride, padding=pads,bias=bias)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride, padding=pads,bias=bias)
        self.conv3 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=ksize, stride=1, padding=ksize//2,bias=bias)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.eps = 1e-5
        # 'FrangiBetaOne', 0.5, 'FrangiBetaTwo', 15
        self.beta = 2 * 0.5 *0.5
        self.c = 2 * 15 * 15

    def forward(self, x):
        pad = self.sigmaK
        Ixx = F.conv2d(x, self.weightxx, padding=pad) * (self.sigma**2)
        Ixy = F.conv2d(x, self.weightxy, padding=pad) * (self.sigma**2)
        Iyy = F.conv2d(x, self.weightyy, padding=pad) * (self.sigma**2)

        diff_xy = Ixx - Iyy
        tmp = torch.sqrt(torch.pow(diff_xy, 2) + 4 * torch.pow(Ixy, 2))
        #Compute the eigenvalues
        mu1 = 0.5 * (Ixx + Iyy + tmp)
        mu2 = 0.5 * (Ixx + Iyy - tmp)

        mu1 = self.norm1(self.conv1(mu1))
        x = self.norm2(self.conv2(x))

        mu1 = torch.cat([mu1, x], dim=1)
        conv3 = self.norm3(self.conv3(mu1))

        # mu1_numpy = mu1.detach().cpu().numpy().astype(np.float32)
        # mu2_numpy = x.detach().cpu().numpy()
        # for i in range(mu1.shape[0]):
        #     label2D = mu1_numpy[i,1, :, :]
        #     img = mu2_numpy[i,1, :, :]
        #     cv2.namedWindow("label2D", cv2.WINDOW_NORMAL)
        #     cv2.imshow("label2D", label2D)
        #     cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        #     cv2.imshow("img", img)
        #
        #     cv2.waitKey(0)

        return conv3

class HessianLayerU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, bias=False, sigma=5):
        super().__init__()

        self.sigmaK = sigma
        self.sigma = sigma/3
        self.H_arange = torch.arange(-self.sigmaK, self.sigmaK + 1)
        self.W_arange = torch.arange(-self.sigmaK, self.sigmaK + 1)
        self.X, self.Y = torch.meshgrid(self.W_arange, self.H_arange, indexing='ij')
        self.X2 = torch.pow(self.X, 2)
        self.Y2 = torch.pow(self.Y, 2)
        self.DGaxx = 1 / (2 * torch.pi * self.sigma ** 4) * (self.X2 / (self.sigma ** 2) - 1) * torch.exp(
            -(self.X2 + self.Y2) / (2 * (self.sigma ** 2)))
        self.DGaxy = 1 / (2 * torch.pi * self.sigma ** 6) * (self.X * self.Y) * torch.exp(
            -(self.X2 + self.Y2) / (2 * (self.sigma ** 2)))
        self.DGayy = torch.t(self.DGaxx)

        self.ksize = self.sigmaK*2+1
        self.weightxx = nn.Parameter(data=self.DGaxx, requires_grad=False).view(1, 1, self.ksize, self.ksize).expand(in_channels, in_channels, self.ksize, self.ksize).cuda().float()
        self.weightxy = nn.Parameter(data=self.DGaxy, requires_grad=False).view(1, 1, self.ksize, self.ksize).expand(in_channels, in_channels, self.ksize, self.ksize).cuda().float()
        self.weightyy = nn.Parameter(data=self.DGayy, requires_grad=False).view(1, 1, self.ksize, self.ksize).expand(in_channels, in_channels, self.ksize, self.ksize).cuda().float()

        self.param = nn.Parameter(torch.ones(1, out_channels, 1, 1),requires_grad=True)  # fixed sin-cos embedding

        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride, padding=0,bias=bias)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride, padding=0,bias=bias)

        self.conv3 = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,bias=bias)

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.eps = -1e4
        # 'FrangiBetaOne', 0.5, 'FrangiBetaTwo', 15
        self.beta = 2 * 0.5 *0.5
        self.c = 2 * 15 * 15

    def forward(self, x):

        pad = self.sigmaK
        Ixx = F.conv2d(x, self.weightxx, padding=pad) * (self.sigma**2)
        Ixy = F.conv2d(x, self.weightxy, padding=pad) * (self.sigma**2)
        Iyy = F.conv2d(x, self.weightyy, padding=pad) * (self.sigma**2)

        diff_xy = Ixx - Iyy
        tmp = torch.sqrt(torch.pow(diff_xy, 2) + 4 * torch.pow(Ixy, 2))
        #Compute the eigenvalues
        mu1 = 0.5 * (Ixx + Iyy + tmp)
        mu2 = 0.5 * (Ixx + Iyy - tmp)

        mu1 = self.norm1(self.conv1(mu1))
        x = self.norm2(self.conv2(x))

        mu1 = torch.cat([mu1, x], dim=1)
        conv3 = self.norm3(self.conv3(mu1))

        return conv3

class HessianNet(nn.Module):

    def __init__(self, in_channel, out_channels=2):

        super().__init__()

        self.conv1 = HessianLayerD(in_channels=in_channel, out_channels=16, ksize=3, stride=1, bias=False, sigma=11)

        self.conv2d = HessianLayerD(in_channels=16, out_channels=32, ksize=3, stride=2, bias=False, sigma=7)

        self.conv3d = HessianLayerD(in_channels=32, out_channels=64, ksize=3, stride=2, bias=False, sigma=5)

        self.conv4d = HessianLayerD(in_channels=64, out_channels=128, ksize=3, stride=2, bias=False, sigma=3)

        # self.conv5d = HessianLayerD(in_channels=128, out_channels=256, ksize=3, stride=2, bias=False, sigma=3)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        # self.conv8u = HessianLayerU(in_channels=256, out_channels=128, ksize=2, stride=2, bias=False, sigma=5)
        # self.conv9 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv10u = HessianLayerU(in_channels=128, out_channels=64, ksize=2, stride=2, bias=False, sigma=5)
        self.conv11 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv12u = HessianLayerU(in_channels=64, out_channels=32, ksize=2, stride=2, bias=False, sigma=7)
        self.conv13 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv14u = HessianLayerU(in_channels=32, out_channels=16, ksize=2, stride=2, bias=False, sigma=11)
        self.conv15 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv16 = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):

        conv1 = self.conv1(x) #16

        conv2d = self.conv2d(conv1)#32

        conv3d = self.conv3d(conv2d)#64

        conv4d = self.conv4d(conv3d)#128

        # conv5d = self.conv5d(conv4d)#256

        conv6 = self.conv6(conv4d)#256
        conv7 = self.conv7(conv6)#256

        # conv8u = self.conv8u(conv7)#128
        # conv8u = torch.cat([conv8u, conv4d], dim=1)
        # conv9 = self.conv9(conv8u) #128

        conv10u = self.conv10u(conv7)#128
        conv10u = torch.cat([conv10u, conv3d], dim=1)
        conv11 = self.conv11(conv10u) #128

        conv12u = self.conv12u(conv11)#128
        conv12u = torch.cat([conv12u, conv2d], dim=1)
        conv13 = self.conv13(conv12u) #128

        conv14u = self.conv14u(conv13)#128
        conv14u = torch.cat([conv14u, conv1], dim=1)
        conv15 = self.conv15(conv14u) #128

        conv16 = self.conv16(conv15) #128

        return conv16, conv16, conv16







