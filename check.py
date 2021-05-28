from scipy import ndimage
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from libs.hardmseg.hardmseg import HarDMSEG
from utils.data_loader import get_loader, ValidationDataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import configparser

# config = configparser.ConfigParser()
# config.read('config.ini')
# optimizer_type = config['Parameter']['optimizer']
# lr = float(config['Parameter']['lr'])
# batch_size = int(config['Parameter']['batch_size'])
# epochs = int(config['Parameter']['epochs'])
# train_size = int(config['Parameter']['train_size'])
# augumentation = config['Parameter'].getboolean('augumentation')
# clip = float(config['Parameter']['clip'])
# train_path = config['Paths']['train_path']
# test_path = config['Paths']['test_path']

# print(optimizer_type, lr, batch_size, epochs, train_size, augumentation, clip, train_path, test_path)


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

from libs.efficientnetv2.effnetv2_cpd import EffNetV2SCPD
from libs.hardmseg.hardmseg import HarDCPD
from libs.efficientnetv2.effnetv2 import effnetv2_s
from libs.hardmseg.hardmseg import hardnet
from libs.pranet.pranet_resnet import CRANet
from torch import nn

# print(dir(nn))


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

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
from libs.respd import ResNetCPD

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

m = nn.AdaptiveAvgPool2d((1,1))
tensor = torch.randn([1, 1024, 11, 11])
out = m(tensor)
print(out.shape)