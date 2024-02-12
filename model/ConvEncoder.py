import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None
        
        if padding:
            if padding is True:
                padding = [int((k-1)/2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1,1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        
        x = self.point_conv(x)

        return x

class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super(MixDropout, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)
    
    def forward(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), kernel=3, activation=nn.ReLU, dropout=0.5):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(num_features=out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, inputs):
        pos = random.randint(1,3)

        x = self.conv1(inputs)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x

class DSCBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.5):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, dropout=0.5):
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=16, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=16, out_c=32, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=128, stride=(2,1), dropout=dropout),
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=128, out_c=128, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=128, out_c=128, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=128, out_c=128, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=128, out_c=256, stride=(1,1), dropout = dropout)
        ])
    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x