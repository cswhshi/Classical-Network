# -*- coding: utf-8 -*-
import torch.nn as nn
#import torch.nn.functional as F
from torchsummary import summary
from LightWeight_Model.Module.ConvBnRelu import ConvBnRelu

"""
depthwise convolution + pointwise convolution+BN+Relu Module
"""
class DWConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,stride, dilation=1, **kwargs):
        super(DWConvBnRelu,self).__init__()
        self.conv = nn.Sequential(
                ConvBnRelu(in_channels,in_channels,kernel_size=3,padding=dilation,stride=stride, dilation=dilation,groups=in_channels),
                ConvBnRelu(in_channels,out_channels,1)
                )
    def forward(self,x):
        return self.conv(x)
    
if __name__ == "__main__":
    dwcbr=DWConvBnRelu(in_channels=3,out_channels=6,stride=1,dilation=1)
    print(dwcbr)
    summary(dwcbr, (3, 224, 224))    







