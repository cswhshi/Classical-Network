# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Fire(nn.Module):
    def __init__(self,inplanes,squeeze_planes,expand_planes):
        super(Fire,self).__init__()
        self.squeeze=nn.Sequential(
                nn.Conv2d(inplanes,squeeze_planes,kernel_size=1,stride=1),
                nn.BatchNorm2d(squeeze_planes),
                nn.ReLU(inplace=True))
        self.expand_1x1=nn.Sequential(
                nn.Conv2d(squeeze_planes,expand_planes,kernel_size=1,stride=1),
                nn.BatchNorm2d(expand_planes))
        self.expand_3x3=nn.Sequential(
                nn.Conv2d(squeeze_planes,expand_planes,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(expand_planes))
        
    def forward(self,x):
        x=self.squeeze(x)
        x1=self.expand_1x1(x)
        x2=self.expand_3x3(x)
        
        x=torch.cat([x1,x2],1)
        x=F.relu(x)
        return x
"""   
from torchsummary import summary   
fire=Fire(64,128,128)
print(fire)
summary(fire, (64,16,16))

"""

class SqueezeNet(nn.Module):
    def __init__(self,classes_num=10):
        super(SqueezeNet,self).__init__()
        self.layer1=nn.Sequential(
                nn.Conv2d(3,96,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2))
        self.fire1=Fire(96,16,64)
        self.fire2=Fire(128,16,64)
        self.fire3=Fire(128,32,128)
        
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fire4=Fire(256, 32, 128)
        self.fire5=Fire(256, 48, 192)
        self.fire6=Fire(384, 48, 192)
        self.fire7=Fire(384, 64, 256)
        
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.fire8=Fire(512,64,256)
        
        self.conv=nn.Conv2d(512,classes_num,kernel_size=1,stride=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x=self.layer1(x)
        x=self.fire1(x)
        x=self.fire2(x)
        x=self.fire3(x)
        x=self.maxpool1(x)
        x=self.fire4(x)
        x=self.fire5(x)
        x=self.fire6(x)
        x=self.fire7(x)
        x=self.maxpool2(x)
        x=self.fire8(x)
        x=self.conv(x)
        x=self.avg_pool(x)
        x=self.softmax(x)
        
        return x

if __name__ == '__main__':     
    net = SqueezeNet()  
    inp =torch.randn(64,3,32,32)
    out = net.forward(inp)
    out.size()
    print(net)
