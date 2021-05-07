"""
EfficientNetV2-S:  params: 24M, FLOPs: 8.8B
source: https://github.com/d-li14/efficientnetv2.pytorch
"""

import torch.nn as nn
import math
import torch

__all__ = ['effnetv2_s']

if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

def _make_divisible(v, divisor, min_value=None):
    """Ensure that all layers have a channel number % 8 = 0
    """
    if min_value is None:
        min_value = divisor
        
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        Swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        Swish()
    )

class SELayer(nn.Module):
    """Squeeze and exciation block: give weightage for each channel
    """
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            Swish(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = [
            #t,  c,  n,  s, SE 
            [1, 24,  2,  1, 0],
            [4, 48,  4,  2, 0],
            [4, 64,  4,  2, 0],
            [4, 128, 6,  2, 1],
            [6, 160, 9,  1, 1],
            [6, 272, 15, 2, 1],]

        input_channel = _make_divisible(24 * width_mult, 8)
        self.layers = [conv_3x3_bn(3, input_channel, 2)]
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                self.layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        # features
        self.features = nn.Sequential(*(self.layers))
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x_medical = x
        out_branch = []
        for i in range(len(self.layers) - 1):
            x_medical = self.layers[i](x_medical)
            if i == 10 or i == 25:
                out_branch.append(x_medical)
        
        out_branch.append(x_medical)
        # print(out_branch[0].shape, out_branch[1].shape, out_branch[2].shape)
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return out_branch

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

def effnetv2_s(pretrained=False, **kwargs):
    """Contructors a EfficientNetV2 model
    """
    model = EffNetV2(**kwargs)
    if pretrained:
        weights = torch.load('./libs/weights/efficientnetv2_s.pth')
        model.load_state_dict(weights)
        print("pretrained EfficientNetV2_S loaded ready")
    return model