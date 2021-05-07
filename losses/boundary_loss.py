import torch
from torch import nn

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

class BDLoss(nn.Module):
    def __init__(self):
        super(BDLoss, self).__init__()
    
    def forward(self, net_output, target, bound):
        """
        net_output (batch_size, class, w,h): predict
        target (batch_size, 1, w,h): ground_truth
        bound (batch_size, class, w,h): precomputed distance map
        """
        net_output = softmax_helper(net_output)
        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:, 1:, ...].type(torch.float32)

        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)
        bd_loss = multipled.mean()
        return bd_loss