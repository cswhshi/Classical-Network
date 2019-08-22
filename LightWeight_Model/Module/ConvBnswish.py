# -*- coding: utf-8 -*-
import torch.nn as nn
from LightWeight_Model.Module import Hswish
from torchsummary import summary
"""
Conv-Bn-Hswish moduule
"""

class ConvBNswish(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                 padding=0,dilation=1,groups=1):
        super(ConvBNswish,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups, bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=Hswish(True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
if __name__ == "__main__":
    convBS=ConvBNswish(in_channels=3,out_channels=32,kernel_size=3,padding=1)
    print(convBS)
    summary(convBS, (3, 224, 224))
    
    


