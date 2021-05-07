import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        # check smooth
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")
    
    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        # target
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # alpha
        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')
        if alpha.device != logit.device:
            alpha.to(logit.device)
        # one_hot_key
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key.to(logit.device)
        # smooth -> one_hot_key
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth/(num_class-1), 1.0-self.smooth)
        # pt
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()
        # gamma
        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1-pt), gamma) * logpt
        # loss according mean() else sum()
        loss = loss.mean() if self.size_average else loss.sum()
        return loss