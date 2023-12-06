import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import config.config as cfg

class pooling2D(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(pooling2D, self).__init__()
        self.pooling_ii = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding = padding)

    def forward(self, x):
        x = self.pooling_ii(x)
        return x
class convLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(convLeaky, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        nn.init.xavier_normal_(self.conv.weight.data)
        # nn.init.constant_(self.conv.bias.data, 0.0)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class convTLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(convTLeaky, self).__init__()
        padding = 0
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        nn.init.xavier_normal_(self.conv.weight.data)
        # nn.init.constant_(self.conv.bias.data, 0.0)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class BaseBlock(nn.Module):

    def __init__(self, channels):
        super(BaseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.netconv1 = convLeaky(channels, channels//2 , 1)
        self.netconv2 = nn.Conv2d(channels//2, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        bn1 = self.bn1(x)
        relu = self.relu(bn1)
        netconv1 = self.netconv1(relu)
        netconv2 = self.netconv2(netconv1)

        out = torch.add(x, netconv2)
        return out


class BLOCK(nn.Module):

    def __init__(self, channels, out_channels):
        super(BLOCK, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.netconv1 = convLeaky(channels, channels, 3)
        self.netconv2 = nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        bn1 = self.bn1(x)
        relu = self.relu(bn1)
        netconv1 = self.netconv1(relu)
        netconv2 = self.netconv2(netconv1)

        return netconv2

class subsNet(nn.Module):

    def __init__(self, inchannel=3, outchaneel=16):
        super(subsNet, self).__init__()


        self.netconv1 = convLeaky(inchannel, 32, 3)
        self.netconv2 = convLeaky(32, 32, 1)
        self.pool1= pooling2D(2, 2, 0)

        self.netconv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = BaseBlock(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.pool2 = pooling2D(2, 2, 0)

        self.netconv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.layer2 = BaseBlock(128)
        self.bn14 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU(0.1, inplace=True)
        self.pool3 = pooling2D(2, 2, 0)

        self.netconv5 = convLeaky(128, 256, 3)
        self.netconv6 = nn.Conv2d(256, 128, 1, stride=1, padding=0, bias=False)
        self.upsample1 = convTLeaky(128, 128, kernel_size=2, stride=2)

        self.block1 = BLOCK(128, 64)
        self.upsample2 = convTLeaky(64, 64, kernel_size=2, stride=2)

        self.block2 = BLOCK(64, 32)
        self.upsample3 = convTLeaky(32, 32, kernel_size=2, stride=2)

        self.block3 = BLOCK(32, 16)

    def forward(self, x):

        netconv1 = self.netconv1(x)
        netconv2 = self.netconv2(netconv1)
        pool1 = self.pool1(netconv2)

        netconv3 = self.netconv3(pool1)
        layer1 = self.layer1(netconv3)
        bn2 = self.bn2(layer1)
        relu2 = self.relu2(bn2)
        pool2 = self.pool2(relu2)

        netconv4 = self.netconv4(pool2)
        layer2 = self.layer2(netconv4)
        bn14 = self.bn14(layer2)
        relu4 = self.relu4(bn14)
        pool3 = self.pool3(relu4)

        ##up_sampleing
        netconv5 = self.netconv5(pool3)
        netconv6 = self.netconv6(netconv5)
        upsample1 = self.upsample1(netconv6)

        cat1 = torch.add(upsample1, layer2)

        block1 = self.block1(cat1)
        upsample2 = self.upsample2(block1)

        cat2 = torch.add(upsample2, layer1)

        block2 = self.block2(cat2)
        block2 = torch.add(block2, pool1)
        upsample3 = self.upsample3(block2)

        cat3 = torch.add(upsample3, netconv1)
        out = self.block3(cat3)

        return out


