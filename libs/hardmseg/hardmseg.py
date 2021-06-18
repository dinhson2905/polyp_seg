import torch
import torch.nn as nn
import torch.nn.functional as F
from .hardnet import hardnet
from .holistic_attention import HA

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
        # x3 have resolution is highest
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

class HarDCPD(nn.Module):
    def __init__(self, channel=32):
        super(HarDCPD, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # RFB module
        self.rfb2_1 = RFB(320, channel)
        self.rfb3_1 = RFB(640, channel)
        self.rfb4_1 = RFB(1024, channel)
        
        self.agg1 = Aggregation(32)
        self.hardnet = hardnet(arch=68)
        self.HA = HA()
        self.rfb2_2 = RFB(320, channel)
        self.rfb3_2 = RFB(640, channel)
        self.rfb4_2 = RFB(1024, channel)
        self.agg2 = Aggregation(32)
    
    def forward(self, x):
        # print(x.size())
        hardnet_out = self.hardnet(x)
        x1 = hardnet_out[0]
        # x2 is optimization layer, x3, x4 last layer conv
        x2 = hardnet_out[1]
        x3 = hardnet_out[2]
        x4 = hardnet_out[3]
        # Branch 1, output of RFB_1
        x2_1 = self.rfb2_1(x2)
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4)
        attention = self.agg1(x4_1, x3_1, x2_1)

        # Branch 2
        x2_2 = self.HA(attention.sigmoid(), x2)
        # x3_2 is output of layer 12 of hardnet
        x3_2 = self.hardnet.base[10](x2_2)
        x3_2 = self.hardnet.base[11](x3_2)
        x3_2 = self.hardnet.base[12](x3_2)
        # x4_2 is output of layer 15 of hardnet
        x4_2 = self.hardnet.base[13](x3_2)
        x4_2 = self.hardnet.base[14](x4_2)
        x4_2 = self.hardnet.base[15](x4_2)
        # output of RFB_2
        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        detection = self.agg2(x4_2, x3_2, x2_2)

        # x2_rfb = self.rfb2_1(x2) # channel -> 32
        # x3_rfb = self.rfb3_1(x3) # channel -> 32
        # x4_rfb = self.rfb4_1(x4) # channel -> 32
        # ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        # lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear') # (bs, 1, 44, 44) -> (bs, 1, 44*8=352, 352)
        attention_up = F.interpolate(attention, scale_factor=8, mode='bilinear')
        detection_up = F.interpolate(detection, scale_factor=8, mode='bilinear')

        # return lateral_map_5
        return attention_up, detection_up

class HarDMSEG(nn.Module):
    def __init__(self, channel=32):
        super(HarDMSEG, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # RFB module
        self.rfb2_1 = RFB(320, channel)
        self.rfb3_1 = RFB(640, channel)
        self.rfb4_1 = RFB(1024, channel)
        
        self.agg1 = Aggregation(32)

        # self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        # self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)

        # self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        # self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        # self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # self.conv2 = BasicConv2d(320, 32, kernel_size=1)
        # self.conv3 = BasicConv2d(640, 32, kernel_size=1)
        # self.conv4 = BasicConv2d(1024, 32, kernel_size=1)
        # self.conv5 = BasicConv2d(1024, 1024, 3, padding=1)
        # self.conv6 = nn.Conv2d(1024, 1, 1)

        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.hardnet = hardnet(arch=68)
    
    def forward(self, x):
        # print(x.size())
        hardnet_out = self.hardnet(x)
        # x1 = hardnet_out[0]
        # x2 is optimization layer, x3, x4 last layer conv
        x2 = hardnet_out[1]
        x3 = hardnet_out[2]
        x4 = hardnet_out[3]
        
        x2_rfb = self.rfb2_1(x2) # channel -> 32
        x3_rfb = self.rfb3_1(x3) # channel -> 32
        x4_rfb = self.rfb4_1(x4) # channel -> 32
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear') # (bs, 1, 44, 44) -> (bs, 1, 44*8=352, 352)
        
        return lateral_map_5

class Attention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention, self).__init__()
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

class HarDPD(nn.Module):
    def __init__(self, channel=32):
        super(HarDPD, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # RFB module
        self.rfb2_1 = RFB(320, channel)
        self.rfb3_1 = RFB(640, channel)
        self.rfb4_1 = RFB(1024, channel)
        
        self.agg1 = Aggregation(32)

        self.hardnet = hardnet(arch=68)
    
    def forward(self, x):
        # print(x.size())
        hardnet_out = self.hardnet(x)
        # x1 = hardnet_out[0]
        # x2 is optimization layer, x3, x4 last layer conv
        x2 = hardnet_out[1]
        x3 = hardnet_out[2]
        x4 = hardnet_out[3]
        
        x2_rfb = self.rfb2_1(x2) # channel -> 32
        x3_rfb = self.rfb3_1(x3) # channel -> 32
        x4_rfb = self.rfb4_1(x4) # channel -> 32
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear') # (bs, 1, 44, 44) -> (bs, 1, 44*8=352, 352)
        
        return lateral_map_5