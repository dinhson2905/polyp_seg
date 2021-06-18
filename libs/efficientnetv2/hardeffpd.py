import torch
import torch.nn as nn
import torch.nn.functional as F
from .effnetv2 import effnetv2_s
from .holistic_attention import HA
from .hardnet import hardnet

if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=5, dilation=5)    
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0,3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        
        return x

class Aggregation(nn.Module):
    def __init__(self, channel):
        super(Aggregation, self).__init__()
        self.relu = nn.ReLU(inplace=True)     
        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)       
        # Concat
        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1

        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel, stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))
    
    def forward(self, x):
        return super().forward(x)

class HarDEffPD(nn.Module):
    def __init__(self, channel=32):
        super(HarDEffPD, self).__init__()
        self.effnetv2_s = effnetv2_s()
        self.hardnet = hardnet(arch=68)
        self.rfb2 = RFB(256, channel)
        self.rfb3 = RFB(512, channel)
        self.rfb4 = RFB(1024, channel)

        self.agg = Aggregation(channel)

        # self.conv1 = ConvLayer(384, 256)
        # self.conv2 = ConvLayer(780, 512)
        # self.conv3 = ConvLayer(1296, 1024)
    
    def forward(self, x):
        x1 = x
        x2 = x
        hardnet_out = self.hardnet(x1)
        x1_1 = hardnet_out[1]
        x1_2 = hardnet_out[2]
        x1_3 = hardnet_out[3]
        x2 = self.effnetv2_s.features(x2)
        x2 = self.effnetv2_s.conv(x2)
        x2_1 = F.interpolate(x2, scale_factor=4, mode='bilinear')
        x2_2 = F.interpolate(x2, scale_factor=2, mode='bilinear')

        # for i in range(len(self.effnetv2_s.layers)):
        #     x2 = self.effnetv2_s.layers[i](x2)
        #     if i == 10:
        #         x2_1 = x2
        #     elif i == 25:
        #         x2_2 = x2
        # x2_3 = x2

        x_1 = torch.cat((x1_1, x2_1), dim=1)
        x_2 = torch.cat((x1_2, x2_2), dim=1)
        x_3 = torch.cat((x1_3, x2), dim=1)
        
        x_1 = self.conv1(x_1)
        x_2 = self.conv1(x_2)
        x_3 = self.conv1(x_3)

        x_1 = self.rfb2(x_1)
        x_2 = self.rfb3(x_2)
        x_3 = self.rfb4(x_3)
        feature = self.agg(x_3, x_2, x_1)
        res_map = F.interpolate(feature, scale_factor=8, mode='bilinear')
        return res_map

class EffNetV2SPD(nn.Module):
    def __init__(self, channel=32):
        super(EffNetV2SPD, self).__init__()
        self.effnetv2_s = effnetv2_s()

        self.rfb2_1 = RFB(256, channel)
        self.rfb3_1 = RFB(512, channel)
        self.rfb4_1 = RFB(1024, channel)

        self.agg1 = Aggregation(channel)
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True
        self.conv2 = conv_1x1_bn(64, 256)
        self.conv3 = conv_1x1_bn(160, 512)
        self.conv4 = conv_1x1_bn(272, 1024)
    
    def forward(self, x):
        for i in range(len(self.effnetv2_s.layers)):
            x = self.effnetv2_s.layers[i](x)
            if i == 10:
                x2 = x
                x2 = self.conv2(x2)
            elif i == 25:
                x3 = x
                x3 = self.conv3(x3)
        x4 = x
        x4 = self.conv4(x4)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')
        return lateral_map_5