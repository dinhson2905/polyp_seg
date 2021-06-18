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
import time

models = [NestedUNet()]
inputs = torch.randn(1, 3, 352, 352)

for model in models:
    model = model.cuda()
    inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    mean_time = np.mean(time_spent)
    print('Avg execution time (ms): {:.3f}'.format(mean_time))
    fps = 1 / mean_time
    print('FPS: ', fps)