# -*- coding: utf-8 -*-
import torch.nn as nn
#import torch.nn.functional as F
from torchsummary import summary

"""
Conv-Bn-Relu moduule
"""
class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                 padding=0,dilation=1,groups=1,relu6=False):
        super(ConvBnRelu,self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True) 
        
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x
if __name__ == "__main__":
    convBR=ConvBnRelu(in_channels=3,out_channels=32,kernel_size=3,padding=1)
    print(convBR)
    summary(convBR, (3, 224, 224))
