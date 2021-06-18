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
from libs.efficientnetv2 import EffNetV2SPD, effnetv2_s, EffNetV2SCPD
from libs.pranet import PraHarDNet, PraNet
from libs.respd import ResNetCPD, ResNetPD
from libs.swin import SwinCPD
from pthflops import count_ops
from torchvision.models import resnet50
from libs.respd.resnet import Res2Net, res2net50_v1b_26w_4s
from libs.unet import U_Net, NestedUNet

def get_n_params(model):
    pp = 0
    i = 0
    for p in list(model.parameters()):
        nn = 1
        # print(i, p.size())
        i += 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

input_tensor = torch.randn((1, 3, 352, 352)).cuda()

models = [EffNetV2SPD(), HarDMSEG(), HarDCPD(), PraNet(), ResNetPD(), ResNetCPD(), U_Net(), NestedUNet()]
# models = [U_Net(), NestedUNet()]
for model in models:
    model_ = model.cuda()
    tensor = input_tensor.cuda()
    CalParams(model, input_tensor)
    # print(get_n_params(model))