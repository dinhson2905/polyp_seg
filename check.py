from scipy import ndimage
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.data_loader import get_loader, ValidationDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, CalParams
import torch.nn.functional as F
from torch import nn
import numpy as np
import configparser
# import model
from libs.hardmseg import HarDCPD, HarDMSEG
from libs.efficientnetv2 import EffNetV2SPD, EffNetV2SCPD
from libs.pranet import PraHarDNet, PraNet
from libs.respd import ResNetCPD, ResNetPD
from libs.unet import U_Net




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
        print(g1.shape)
        x1 = self.W_x(x)
        print(x1.shape)
        psi = self.relu(g1 + x1)
        print(psi.shape)
        psi = self.psi(psi)

        return x * psi

attention = Attention(32, 32, 16)
x5 = torch.randn(1, 32, 352, 352)
x4 = torch.randn(1, 32, 352, 352)
k = attention(g=x5, x=x4)
d = torch.cat([k, x5], dim=1)
print(d.shape)

# inp = torch.randn(1, 3, 352, 352).cuda()
# model = EffNetV2SCPD().cuda()
# res, res2 = model(inp)
# print(res.shape, res2.shape)

# input_tensor = torch.randn(16, 1, 352, 352).cuda()

# model = HarDMSEG().cuda()

# a, d = model(input_tensor)

# print(a.size(), d.size())
# from losses.region_loss import DiceBCELoss
# from losses.region_loss import WIoUBCELoss
# dice_bce = DiceBCELoss()
# inputs = torch.randn(16, 1, 352, 352)
# targets = torch.ones(16, 1, 352, 352)
# print(inputs.shape, targets.shape)
# res = dice_bce(inputs, targets)
# print(res)


# def struct_loss(pred, mask):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     print(weit.shape)
#     bce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     print(bce.shape)
#     print(bce)
#     wbce = (weit * bce).sum(dim=(2,3)) / weit.sum(dim=(2,3))
#     # print(wbce.shape)
#     # print(wbce)
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2,3))
#     union = ((pred + mask) * weit).sum(dim=(2,3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     print(wbce.shape, wiou.shape)
#     return (wbce + wiou).mean(), (bce + wiou).mean()


# inputs = torch.randn(1, 1, 352, 352)
# targets = torch.ones(1, 1, 352, 352)
# wiou_bce = WIoUBCELoss()
# # weight = torch.abs(F.avg_pool2d(test_tensor, kernel_size=3, stride=1, padding=1) - test_tensor)
# out, out2 = struct_loss(inputs, targets)
# res = wiou_bce(inputs, targets)
# print(out, out2, res)


# out = struct_loss(inputs, targets)
# print(out.shape)


# def struct_loss2(pred, mask):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     print(weit.shape)


#     weit = weit.view(-1)
#     pred = torch.sigmoid(pred)
#     pred = pred.view(-1)
#     mask = mask.view(-1)

#     wbce = F.binary_cross_entropy(pred, mask, reduction='mean')
#     wbce = (weit * wbce).sum() / weit.sum()
#     print(wbce)
#     inter = ((pred * mask) * weit).sum()
#     union = ((pred + mask) * weit).sum()
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     print(wbce.shape, wiou.shape)
#     return (wbce + wiou).mean()

# out = struct_loss(inputs, targets)
# out_2 = struct_loss2(inputs, targets)
# print(out, out_2)

# from sklearn.metrics import f1_score, recall_score, precision_score

# y_true = [0.0, 1.0, 0.9999, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
# y_pred = [0.0, 1.0, 0.0, 0.9998, 1.0, 0.0, 1.0, 0.0002, 0.0, 0.0, 0.0]
# def dice_coeff(inputs, targets, smooth=1):
#     inputs_flat = np.reshape(inputs, -1)
#     targets_flat = np.reshape(targets, -1)
#     intersection = (inputs_flat * targets_flat)
#     dice_score = (2 * intersection.sum() + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
#     return dice_score

# targets = np.array(y_true).astype(int)
# inputs = np.array(y_pred).astype(int)
# print(targets, inputs   )

# recall = recall_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred)
# print(recall, precision)

# from libs.pranet.pranet_resnet import CRANet

# model = CRANet().cuda()
# input_tensor = torch.randn(1, 3, 352, 352).cuda()
# out = model(input_tensor)
# print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
# print(dir(nn))




# from libs.swin.swincpd import SwinCPD

# model = SwinCPD().cuda()
# print(get_n_params(model))
# tensor = torch.randn([8, 3, 352, 352]).cuda()
# res = model(tensor)
# print(res.shape)

# model = EffNetV2SCPD().cuda()
# tensor = torch.randn([1, 3, 352, 352]).cuda()
# res = model(tensor)
# print(res.shape)

# model = ResNetCPD().cuda()
# print(get_n_params(model))
# tensor = torch.randn([4, 3, 352, 352]).cuda()
# res1, res2 = model(tensor)
# print(res2.shape, res1.shape)

# from libs.respd import ResNetPD

# model = ResNetPD().cuda()
# print(get_n_params(model))
# tensor = torch.randn([8, 3, 352, 352]).cuda()
# res2 = model(tensor)
# print(res2.shape)

# m = nn.AdaptiveAvgPool2d((1,1))
# tensor = torch.randn([1, 1024, 11, 11])
# out = m(tensor)
# print(out.shape)

