# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from LightWeight_Model.Module.ConvBnRelu import ConvBnRelu
import torch.nn.functional as F
from torchsummary import summary

def channel_shuffle(x,groups):
    n,c,h,w=x.size()
    channels_per_group=c//groups
    x = x.view(n, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(n, -1, h, w)
    return x

class ShuffleNetUtil(nn.Module):
    def __init__(self,in_channels,out_channels,stride,groups,dilation=1):
        super(ShuffleNetUtil,self).__init__()
        self.stride=stride   #步幅
        self.groups=groups   #分组
        self.dilation=dilation    #空洞率
        assert stride in [1,2,3]           #
        inter_channels=out_channels//4     #瓶颈通道的数量
        if(stride>1):
            self.shortcut = nn.AvgPool2d(3, stride, 1)
            out_channels -= in_channels
        elif(dilation>1):
            out_channels-=in_channels
        if in_channels==24:
            g=1 
        else:
            g=groups
        
        self.conv1=ConvBnRelu(in_channels,inter_channels,1,groups=g)
        self.conv2=ConvBnRelu(inter_channels,inter_channels,3,stride,dilation,dilation,groups)
        self.conv3=nn.Sequential(
                nn.Conv2d(inter_channels, out_channels, 1, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels))
    def forward(self,x):
        out=self.conv1(x)                      #分组卷积
        out=channel_shuffle(out,self.groups)   #channle shuffle
        out=self.conv2(out)                    #深度可分卷积
        out=self.conv3(out)                    #分组卷积，不进行激活
        if self.stride > 1:
            x = self.shortcut(x)               #下采样功能的 ShuffleNet unit中，需要将x的分辨率降低，然后才能concat连接，使用concat连接这样做的目的主要是降低计算量与参数大小
            out = torch.cat([out, x], dim=1)
        elif self.dilation > 1:                #如果采用空洞卷积的话，也是采用concat的连接方式
            out = torch.cat([out, x], dim=1)
        else:                                   #残差块里面如果没有降低分辨率的，连接方式使用加操作(Add)
            out = out + x
        out = F.relu(out)
        return out



class ShuffleNet(nn.Module):
    def __init__(self,num_classes=10,groups=8,dilated=False):
        super(ShuffleNet,self).__init__()
        if(groups==1):
            stages_out_channels=[144,288,576]
        elif groups == 2:
            stages_out_channels = [200, 400, 800]
        elif groups == 3:
            stages_out_channels = [240, 480, 960]
        elif groups == 4:
            stages_out_channels = [272, 544, 1088]
        elif groups == 8:
            stages_out_channels = [384, 768, 1536]
        else:
            raise ValueError("Unknown groups.")

        stages_repeats=[3,7,3]  #对应分辨率下的shuffle util重复次数
        
        self.conv1=ConvBnRelu(3,24,kernel_size=3,stride=2,padding=1)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.in_channels = 24
        
        self.stage1=self.make_stage(stages_out_channels[0],stages_repeats[0],groups)
        if(dilated):#如果使用了空洞卷积，分辨率将保持不变
            self.stage2=self.make_stage(stages_out_channels[1], stages_repeats[1], groups, 2)
            self.stage3=self.make_stage(stages_out_channels[2], stages_repeats[2], groups, 2)
        else:
            self.stage2=self.make_stage(stages_out_channels[1],stages_repeats[1],groups)
            self.stage3=self.make_stage(stages_out_channels[2],stages_repeats[2],groups)
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(stages_out_channels[2],num_classes)
        
    def make_stage(self,out_channels,repeats,groups,dilation=1):
        if(dilation==1):
            stride=2
        else:
            stride=1
        #第一个shuffle util需要传入stride将分辨率降低（没有使用空洞卷积的话）
        layers=[ShuffleNetUtil(self.in_channels,out_channels,stride,groups,dilation)]
        self.in_channels=out_channels
        for i in range(repeats):
            layers.append(ShuffleNetUtil(self.in_channels,out_channels,1, groups))
            self.in_channels=out_channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.avg_pool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

def get_shufflenet(groups):
    return ShuffleNet(groups=groups)
    
"""
input = torch.rand(1,3,224,224)
model=get_shufflenet(8)
print(model)
from torchsummary import summary
summary(model, (3, 224, 224))

"""


