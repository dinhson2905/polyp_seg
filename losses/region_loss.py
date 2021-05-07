import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_averange=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = bce + dice_loss
        return dice_bce

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou

class WIoUBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(WIoUBCELoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weit = weit.view(-1)
        wbce = F.binary_cross_entropy(inputs, targets, weight=weit, reduction='mean')
        intersection = ((inputs * targets) * weit).sum()
        total = ((inputs + targets) * weit).sum()
        union = total - intersection
        wiou = 1 - (intersection + smooth) / (union + smooth)
        return (wbce + wiou).mean()