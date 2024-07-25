"""
This part contains UNet series models, 
including UNet, R2UNet, Attention UNet, R2Attention UNet, DenseUNet
"""
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from timm.models.vision_transformer import Block
from net.pp_lite_seg import  UAFM_SpAtten


# ==========================Core Module================================
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=scale_factor),
            nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):  # attention Gate
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

# =====R2U增=====
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvBlock(nn.Module):

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int],
                 padding: Union[int, None] = None,
                 activation: bool = True,
                 **kwargs):
        super(ConvBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=autopad(k=kernel_size, p=padding), **kwargs)
        self.bn = nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        if self.activation:
            out = F.relu(out)
        return out


class ResPath(nn.Module):

    def __init__(self,
                 input_channels : int,
                 output_channels: int,
                 length: int,
                 padding: Union[int, None] = None):
        super(ResPath, self).__init__()
        self.length = length
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = ConvBlock(input_channels=self.input_channels,
                               output_channels=self.output_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=autopad(k=(1, 1), p=padding),
                               activation=False)

        self.conv2 = ConvBlock(input_channels=self.input_channels,
                               output_channels=self.output_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=autopad(k=(3, 3), p=padding))

        self.bn = nn.BatchNorm2d(num_features=self.conv2.output_channels)

        self.module = nn.ModuleList()

        for i in range(self.length-1):
            self.module.append(module=ConvBlock(input_channels=self.output_channels,
                                                output_channels=self.output_channels,
                                                kernel_size=(1, 1),
                                                stride=(1, 1),
                                                activation=False,
                                                padding=autopad(k=(1, 1), p=padding)))
            self.module.append(module=ConvBlock(input_channels=self.output_channels,
                                                output_channels=self.output_channels,
                                                kernel_size=(3, 3),
                                                stride=(1, 1),
                                                padding=autopad(k=(3, 3), p=padding)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        shortcut = self.conv1(shortcut)
        out = self.conv2(x)
        out = torch.add(shortcut, out)
        out = F.relu(out)
        out = self.bn(out)

        for i in range(self.length-1):
            shortcut = out
            shortcut = self.module[i*2](shortcut)
            out = self.module[i*2+1](out)
            out = torch.add(shortcut, out)
            out = F.relu(out)
            out = self.bn(out)

        return out

# ==================================================================
class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, fea_channels=32):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=fea_channels)
        self.Conv2 = conv_block(ch_in=fea_channels, ch_out=fea_channels*2)
        self.Conv3 = conv_block(ch_in=fea_channels*2, ch_out=fea_channels*4)
        self.Conv4 = conv_block(ch_in=fea_channels*4, ch_out=fea_channels*8)
        self.Conv5 = conv_block(ch_in=fea_channels*8, ch_out=fea_channels*16)

        self.Up5 = up_conv(ch_in=fea_channels*16, ch_out=fea_channels*8)
        self.Up_conv5 = conv_block(ch_in=fea_channels*16, ch_out=fea_channels*8)

        self.Up4 = up_conv(ch_in=fea_channels*8, ch_out=fea_channels*4)
        self.Up_conv4 = conv_block(ch_in=fea_channels*8, ch_out=fea_channels*4)

        self.Up3 = up_conv(ch_in=fea_channels*4, ch_out=fea_channels*2)
        self.Up_conv3 = conv_block(ch_in=fea_channels*4, ch_out=fea_channels*2)

        self.Up2 = up_conv(ch_in=fea_channels*2, ch_out=fea_channels*1)
        self.Up_conv2 = conv_block(ch_in=fea_channels*2, ch_out=fea_channels*1)

        self.Conv_1x1 = nn.Conv2d(fea_channels*1, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        # print('x1',x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print('x2',x2.shape)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # print(x4.shape)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        # print(d4.shape)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return [d1, d1,d1]

from functools import partial
from net.pos_embdb import get_2d_sincos_pos_embed
##########################################################################################
class U_Nett(nn.Module):
    def __init__(self, img_ch=3, output_ch=2, fea_channels=64):
        super(U_Nett, self).__init__()
  
        self.Maxpool1 = nn.Conv2d(fea_channels, fea_channels, kernel_size=3, stride=2, padding =1, bias = False)#nn.MaxPool2d(kernel_size=2, stride=2)#
        self.Maxpool2 = nn.Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=2, padding =1, bias = False)#nn.MaxPool2d(kernel_size=2, stride=2)#
        self.Maxpool3 = nn.Conv2d(fea_channels*4, fea_channels*4, kernel_size=3, stride=2, padding =1, bias = False)#nn.MaxPool2d(kernel_size=2, stride=2)#

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=fea_channels)
        self.Conv2 = conv_block(ch_in=fea_channels, ch_out=fea_channels*2)
        self.Conv3 = conv_block(ch_in=fea_channels*2, ch_out=fea_channels*4)
        self.Conv4 = conv_block(ch_in=fea_channels*4, ch_out=fea_channels*8)

        self.sizem = 24
        self.param = nn.Parameter(torch.zeros(1, self.sizem * self.sizem, 1),requires_grad=True)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.sizem*self.sizem, fea_channels*8), requires_grad=False)  # fixed sin-cos embedding
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.transblock = nn.ModuleList([
            Block(dim=fea_channels*8, num_heads=fea_channels*8//64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
            for i in range(4)])
        
        # self.sizem = 32
        # self.param = nn.Parameter(torch.zeros(1, self.sizem * self.sizem, 1),requires_grad=True)  # fixed sin-cos embedding
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.sizem*self.sizem, fea_channels*8), requires_grad=False)  # fixed sin-cos embedding
        # norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.transblock = nn.ModuleList([
        #     Block(dim=fea_channels*8, num_heads=fea_channels*8//64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(4)])
        
    
        #self.UaSp1 = UAFM_SpAtten(x_ch=fea_channels*4, y_ch=fea_channels*8, out_ch=fea_channels*4, ukernel_size=2, ksize=3)

        self.Up4 = up_conv(ch_in=fea_channels*8, ch_out=fea_channels*4)
        self.Up_conv4 = conv_block(ch_in=fea_channels*8, ch_out=fea_channels*4)

        #self.UaSp2 = UAFM_SpAtten(x_ch=fea_channels * 2, y_ch=fea_channels * 4, out_ch=fea_channels * 4, ukernel_size=2,ksize=3)
        self.Up3 = up_conv(ch_in=fea_channels*4, ch_out=fea_channels*2)
        self.Up_conv3 = conv_block(ch_in=fea_channels*4, ch_out=fea_channels*2)

        #self.UaSp3 = UAFM_SpAtten(x_ch=fea_channels * 1, y_ch=fea_channels * 2, out_ch=fea_channels * 2, ukernel_size=3,ksize=3)
        self.Up2 = up_conv(ch_in=fea_channels*2, ch_out=fea_channels*1, scale_factor=2)
        self.Up_conv2 = conv_block(ch_in=fea_channels*2, ch_out=fea_channels*1)

        self.Conv_1x1 = nn.Conv2d(fea_channels*1, output_ch, kernel_size=1, stride=1, padding=0)
        self.initialize_weights()
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.sizem), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        # print('x1',x1.shape)
        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)
        # print('x2',x2.shape))
        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)
        
        b, c, h, w = x4.shape
        x4 = x4.view(b, c, h* w).permute(0, 2, 1)
        xc = x4+ self.pos_embed
        for cpb in self.transblock:
            xc = cpb(xc)
        x4 = (self.param*xc + x4).permute(0, 2, 1).view(b, c, h, w)
        # x4 = xc.permute(0, 2, 1).view(b, c, h, w)

        # d4 = self.UaSp1(x3, x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        #d3 = self.UaSp2(x2, d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        #d2 = self.UaSp3(x1, d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine
        #d1 = F.sigmoid(d1)  # mine
        
        return [d1, d1,d1]

if __name__ == '__main__':
    net = U_Net(1,2)
    in1 = torch.randn(1,1,64,64)
    out1 = net(in1)
    print(out1.shape)

# # 计算参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# if __name__ == '__main__':
#     net = U_Net(1,2)
#     in1 = torch.randn(1,1,64,64)
#     out1 = net(in1)
#     # print(net)
#     print("Total number of parameters: " + str(count_parameters(net)))
#

# if __name__ == '__main__':
#     net = U_Net(1,2)
#     in1 = torch.randn(1,1,64,64)
#
# if __name__ == '__main__':
#     import time
#     batch_size = 1
#     # batch = torch.zeros([batch_size, 1, 80, 80], dtype=torch.float32)
#     batch = torch.randn(1,1,64,64)
#     model = net
#     print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
#     print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
#     start_time = time.time()
#     logits = model(batch)
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     # 计算FLOPs & Params
#     from models.util import CalParams
#     t = CalParams(net.cuda(), torch.rand(1,1,64,64).cuda())
#     print(t)


# ============================================================
class R2U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.ResPath1 = ResPath(input_channels=img_ch, output_channels=64,length=5)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.ResPath2 = ResPath(input_channels=64,output_channels=128,length=4)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.ResPath3 = ResPath(input_channels=128, output_channels=256,length=3)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)
        self.ResPath4 = ResPath(input_channels=256, output_channels=512,length=2)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)
        self.ResPath5 = ResPath(input_channels=512,output_channels=1024,length=1)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)
        # x1_1 = self.ResPath1(x1)
        #print(x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        # x2_1 = self.ResPath2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        # x3_1 = self.ResPath3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        # x4_1 = self.ResPath4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)
        # x5_1 = self.ResPath5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        # d5 = torch.cat((x4_1, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        # d4 = torch.cat((x3_1, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        # d3 = torch.cat((x2_1, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        # d2 = torch.cat((x1_1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)

        return d1

# if __name__ == '__main__':
#     net = R2U_Net(1,2)
#     print(net)
#     in1 = torch.randn((64,1,48,48)).cuda()
#     out1 = net(in1)
#     print(out1.size())

# ===========================================================
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)
        return d1


# if __name__ == '__main__':
#     net = AttU_Net(1,2)
#     in1 = torch.randn(1,1,64,64)
#
# if __name__ == '__main__':
#     import time
#     batch_size = 1
#     # batch = torch.zeros([batch_size, 1, 80, 80], dtype=torch.float32)
#     batch = torch.randn(1,1,64,64)
#     model = net
#     print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
#     print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
#     start_time = time.time()
#     logits = model(batch)
#     print("--- %s seconds ---" % (time.time() - start_time))
#
#     # 计算FLOPs & Params
#     from models.util import CalParams
#     t = CalParams(net.cuda(), torch.rand(1,1,64,64).cuda())
#     print(t)

# ==============================================================
class R2AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1, dim=1)

        return d1

#==================DenseUNet=====================================
class Single_level_densenet(nn.Module):
    def __init__(self, filters, num_conv=4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, 3, padding=1))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class Down_sample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Down_sample, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        y = self.down_sample_layer(x)
        return y, x


class Upsample_n_Concat(nn.Module):
    def __init__(self, filters):
        super(Upsample_n_Concat, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding=1, stride=2)
        self.conv = nn.Conv2d(2 * filters, filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, x, y):
        x = self.upsample_layer(x)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        return x


class Dense_Unet(nn.Module):
    def __init__(self, in_chan=3,out_chan=2,filters=128, num_conv=4):

        super(Dense_Unet, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, filters, 1)
        self.d1 = Single_level_densenet(filters, num_conv)
        self.down1 = Down_sample()
        self.d2 = Single_level_densenet(filters, num_conv)
        self.down2 = Down_sample()
        self.d3 = Single_level_densenet(filters, num_conv)
        self.down3 = Down_sample()
        self.d4 = Single_level_densenet(filters, num_conv)
        self.down4 = Down_sample()
        self.bottom = Single_level_densenet(filters, num_conv)
        self.up4 = Upsample_n_Concat(filters)
        self.u4 = Single_level_densenet(filters, num_conv)
        self.up3 = Upsample_n_Concat(filters)
        self.u3 = Single_level_densenet(filters, num_conv)
        self.up2 = Upsample_n_Concat(filters)
        self.u2 = Single_level_densenet(filters, num_conv)
        self.up1 = Upsample_n_Concat(filters)
        self.u1 = Single_level_densenet(filters, num_conv)
        self.outconv = nn.Conv2d(filters, out_chan, 1)

    #         self.outconvp1 = nn.Conv2d(filters,out_chan, 1)
    #         self.outconvm1 = nn.Conv2d(filters,out_chan, 1)

    def forward(self, x):
        x = self.conv1(x)
        x, y1 = self.down1(self.d1(x))
        x, y2 = self.down1(self.d2(x))
        x, y3 = self.down1(self.d3(x))
        x, y4 = self.down1(self.d4(x))
        x = self.bottom(x)
        x = self.u4(self.up4(x, y4))
        x = self.u3(self.up3(x, y3))
        x = self.u2(self.up2(x, y2))
        x = self.u1(self.up1(x, y1))
        x1 = self.outconv(x)
        #         xm1 = self.outconvm1(x)
        #         xp1 = self.outconvp1(x)
        x1 = F.softmax(x1,dim=1)
        return x1
# =========================================================

# if __name__ == '__main__':
#     net = Dense_Unet(3,21,128).cuda()
#     print(net)
#     in1 = torch.randn(4,3,224,224).cuda()
#     out = net(in1)
#     print(out.size())

# if __name__ == '__main__':
#     # test network forward
#     net = AttU_Net(1,2).cuda()
#     print(net)
#     in1 = torch.randn((4,1,48,48)).cuda()
#     out1 = net(in1)
#     print(out1.size())

