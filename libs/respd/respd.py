import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
from .holistic_attention import HA
from .resnet import res2net50_v1b_26w_4s

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class ResNetPD(nn.Module):
    def __init__(self, channel=32):
        super(ResNetPD, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.rfb2_1 = RFB(512, channel)
        self.rfb3_1 = RFB(1024, channel)
        self.rfb4_1 = RFB(2048, channel)

        self.agg1 = aggregation(channel)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # (b, 64, 88, 88)
        x1 = self.resnet.layer1(x) # (b, 256, 88, 88)
        
        x2 = self.resnet.layer2(x1) # (b, 512, 44, 44)
        x3 = self.resnet.layer3(x2) # (b, 1024, 22, 22)
        x4 = self.resnet.layer4(x3) # (b, 2048, 11, 11)
        
        x2_rfb = self.rfb2_1(x2) # channel -> 32
        x3_rfb = self.rfb3_1(x3) # channel -> 32
        x4_rfb = self.rfb4_1(x4) # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear') # S-1: (bs, 1, 44, 44) -> (b, 1, 352, 352)

        return lateral_map_5
    
class ResNetCPD(nn.Module):
    def __init__(self, channel=32):
        super(ResNetCPD, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.rfb2 = RFB(512, channel)
        self.rfb3 = RFB(1024, channel)
        self.rfb4 = RFB(2048, channel)

        self.agg = aggregation(channel)
        self.HA = HA()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # (b, 64, 88, 88)
        x1 = self.resnet.layer1(x) # (b, 256, 88, 88)
        
        x2 = self.resnet.layer2(x1) # (b, 512, 44, 44)
        x3 = self.resnet.layer3(x2) # (b, 1024, 22, 22)
        x4 = self.resnet.layer4(x3) # (b, 2048, 11, 11)
        
        x2_1 = self.rfb2(x2) # channel -> 32
        x3_1 = self.rfb3(x3) # channel -> 32
        x4_1 = self.rfb4(x4) # channel -> 32

        attention = self.agg(x4_1, x3_1, x2_1)

        # branch 2
        x2_2 = self.HA(attention.sigmoid(), x2)
        x3_2 = self.resnet.layer3(x2_2)
        x4_2 = self.resnet.layer4(x3_2)

        x2_2 = self.rfb2(x2_2)
        x3_2 = self.rfb3(x3_2)
        x4_2 = self.rfb4(x4_2)
        detection = self.agg(x4_2, x3_2, x2_2)

        attention_map = F.interpolate(attention, scale_factor=8, mode='bilinear')
        detection_map = F.interpolate(detection, scale_factor=8, mode='bilinear')

        return attention_map, detection_map

if __name__ == '__main__':
    ras = ResNetPD().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    out = ras(input_tensor)


