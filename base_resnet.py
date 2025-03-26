# -*- coding: utf-8 -*-

'''
This is full set for cifar-10 datasets 
Models: ResNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from fedrolex.modules import Scaler

############# ResNet #############

class BatchNorm_no_tracking(nn.BatchNorm2d):
    def __init__(self, num_features: int):
        super().__init__(num_features,  track_running_stats=False)

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_groups=2, num_channels=num_channels)

class InstanceNorm(nn.GroupNorm):
    def __init__(self, num_channels: int):
        super().__init__(num_groups=num_channels, num_channels=num_channels)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d, sequential=nn.Sequential, rate=1, cfg=None, tracking=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(planes,  affine=True, track_running_stats=tracking)
        
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes,  affine=True, track_running_stats=tracking)

        if cfg is not None:
            if cfg['shared']['scale']:
                self.scaler = Scaler(rate)
            else:
                self.scaler = nn.Identity()
        else:
                self.scaler = nn.Identity()

        self.shortcut = sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = sequential(
                conv_layer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
            self.shortcut.append(norm_layer(self.expansion*planes,  affine=True, track_running_stats=tracking))
            
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, hidden_sizes=[64, 128, 256, 512], n_class=10, norm_layer=nn.BatchNorm2d, \
                    conv_layer=nn.Conv2d, linear_layer=nn.Linear, sequential=nn.Sequential, rate=1, cfg=None, tracking=False):
        super(ResNet, self).__init__()

        self.in_planes = hidden_sizes[0]

        self.conv1 = conv_layer(3, hidden_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm_layer(hidden_sizes[0],  affine=True, track_running_stats=tracking)
        
        self.layer1 = self._make_layer(block, hidden_sizes[0], num_blocks[0], stride=1, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential, rate=rate, cfg=cfg, tracking=tracking)
        self.layer2 = self._make_layer(block, hidden_sizes[1], num_blocks[1], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential, rate=rate, cfg=cfg, tracking=tracking)
        self.layer3 = self._make_layer(block, hidden_sizes[2], num_blocks[2], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential, rate=rate, cfg=cfg, tracking=tracking)
        self.layer4 = self._make_layer(block, hidden_sizes[3], num_blocks[3], stride=2, norm_layer=norm_layer, conv_layer=conv_layer, sequential=sequential, rate=rate, cfg=cfg, tracking=tracking)
                
        if cfg is not None:
            if cfg['shared']['scale']:
                self.scaler = Scaler(rate)
            else:
                self.scaler = nn.Identity()
        else:
                self.scaler = nn.Identity()

        self.linear = linear_layer(hidden_sizes[3]*block.expansion, n_class)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, conv_layer, sequential, rate, cfg, tracking):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes=planes, stride=stride, norm_layer=norm_layer, rate=rate, cfg=cfg, tracking=tracking))
            self.in_planes = planes * block.expansion
        return sequential(*layers)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out