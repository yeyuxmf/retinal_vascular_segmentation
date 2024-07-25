import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Union
from network_architecture.neural_network import SegmentationNetwork
from network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock
from  net.UNetFamily import U_Nett
from net.dual_local_global_attention import Patch_AttentionV2
class UpSampling(nn.Module):
    def __init__(self, scale_factor =2):
        super(UpSampling, self).__init__()
        self.upsample = nn.Upsample(scale_factor = scale_factor,  mode='bilinear', align_corners=None)

    def forward(self, x):
        x = self.upsample(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, norm_name='batch'):
        super(FeatureFusion, self).__init__()
        self.urb1 = UnetResBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=3, padding=2, bias=False)
        self.conv2 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.urb2 = UnetResBlock(
            spatial_dims=2,
            in_channels=out_channels*2,
            out_channels=out_channels*1,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )


    def forward(self, x, multif):
        b, c, h, w = x.shape
        dec1, dec2 = multif
        urb1 = self.urb1(x)

        conv1 = self.conv1(urb1)
        dec1 = self.conv2(dec1)
        conv1 = torch.cat([conv1, dec1], dim=1)
        urb2 = self.urb2(conv1)



        urb2 = F.interpolate(urb2, (h, w ), mode='bilinear')

        ff = torch.cat([urb1, urb2], dim=1)

        return ff



class RetinalVasularSegFF(SegmentationNetwork):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [640, 640],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (3, 3)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size
        self.input_size = [self.feat_size[0]*8*self.feat_size[1]*8, self.feat_size[0]*4*self.feat_size[1]*4,
                           self.feat_size[0]*2*self.feat_size[1]*2, self.feat_size[0]*1*self.feat_size[1]*1]
        self.unetr_pp_encoder = UnetrPPEncoder(input_size=self.input_size, patch_size=self.patch_size, dims=dims, depths=depths, num_heads=num_heads, in_channels=feature_size)

        self.prarm = nn.Parameter(torch.tensor([0.2, 0.6, 0.2]),requires_grad=True).cuda().float()
        norm_name ='batch'
        self.unet = U_Nett(img_ch=feature_size, output_ch=feature_size, fea_channels=32)
        self.encoder1 = UnetResBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.la1 = UnetResBlock(
            spatial_dims=2,
            in_channels=feature_size*2,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.la2 = UnetResBlock(
            spatial_dims=2,
            in_channels=feature_size*4,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.la3 = UnetResBlock(
            spatial_dims=2,
            in_channels=feature_size*8,
            out_channels=feature_size*8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )



        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=img_size[0]//(self.patch_size[0]*4) * img_size[1]//(self.patch_size[1]*4),
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=img_size[0]//(self.patch_size[0]*2) * img_size[1]//(self.patch_size[1]*2),
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=img_size[0]//self.patch_size[0] * img_size[1]//self.patch_size[1],
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=self.patch_size,
            norm_name=norm_name,
            out_size=img_size[0] * img_size[1],
            conv_decoder=True,
        )
        # self.FeaFus = FeatureFusion(feature_size, feature_size)
        self.out1 = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)
        # if self.do_ds:
        #     self.out2 = UnetOutBlock(spatial_dims=2, in_channels=feature_size * 2, out_channels=out_channels)
        #     self.out3 = UnetOutBlock(spatial_dims=2, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in):

        convBlock = self.encoder1(x_in)
        x_output, hidden_states = self.unetr_pp_encoder(convBlock)


        # convBlock = self.encoder11(convBlock)
        convBlock1 = self.unet(convBlock)[0]

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        enc1 = self.la1(enc1)
        enc2 = self.la2(enc2)
        enc3 = self.la3(enc3)

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        decoder2 = self.decoder2(dec1, convBlock1)
        out = decoder2#self.FeaFus(decoder2, [dec1, dec2])
        # if self.do_ds:
        #     logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        # else:
        logits = self.out1(out)

        return logits, logits, logits
