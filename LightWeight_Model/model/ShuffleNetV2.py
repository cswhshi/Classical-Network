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

class DWConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0, dilation=1,bias=False):
        super(DWConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,
                              padding, dilation, groups=in_channels, bias=bias)
    def forward(self,x):
        return self.conv(x)
    
class ShuffleNetV2Util(nn.Module):
    """
    有效的网络架构设计推导出的一些实用指南：
    （Ⅰ）G1:相等的通道宽度可最大限度地降低内存访问成本（MAC）； 
    （Ⅱ）G2:过多的组卷积会增加MAC； 
    （Ⅲ）G3:网络碎片降低了并行度； 
    （Ⅳ）G4:逐元素操作的执行时间是不可忽略的； 
    基于上述指导原则和研究，有效的网络架构应该： 
    （Ⅰ）使用“平衡卷积"（相等的通道宽度）; 
    （Ⅱ）注意使用组卷积的成本; 
    （Ⅲ）降低碎片程度; 
    （Ⅳ）减少逐元素操作。
    """
    def __init__(self,in_channels,out_channels,stride,dilation=1):
        super(ShuffleNetV2Util,self).__init__()
        assert stride in [1,2,3]
        self.stride = stride
        self.dilation = dilation
        
        inter_channels = out_channels // 2   #channel split
        
        if(stride>1 or dilation>1):#带下采样的模块，左边的路径的特征图也需要进行相应的下采样，同时也不使用channel split
            self.branch1=nn.Sequential(
                    DWConv(in_channels,in_channels,3,stride, dilation, dilation),
                    nn.BatchNorm2d(in_channels),
                    ConvBnRelu(in_channels,inter_channels,1))
        self.branch2=nn.Sequential(#如果带下采样的模块，右侧的路径有所不同，也就是不需要进行channel split
                ConvBnRelu(in_channels if (stride > 1) else inter_channels,inter_channels, 1),
                DWConv(inter_channels,inter_channels,3,stride,dilation, dilation),
                nn.BatchNorm2d(inter_channels),
                ConvBnRelu(inter_channels,inter_channels,1)
                )
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self,x):
        if(self.stride==1 and self.dilation==1):#如果不进行下采样，则左路不需要做任何运算
            x1,x2=x.chunk(2,dim=1)#torch.chunk（input, chunks, dim），与torch.cat()的作用相反。注意，返回值的数量会随chunks的值而发生变化.
            out=torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out=torch.cat((self.branch1(x), self.branch2(x)),dim=1)
        out=channel_shuffle(out,2)#参数2表示groups为2组，因为分成两条路径，生成两组特征图
        return out
    

#if __name__ == "__main__":
"""
V2Util=ShuffleNetV2Util(64,64,2,1)
print(V2Util)
summary(V2Util, (64, 27, 27))    
"""
 

class ShuffleNetV2(nn.Module):
    """
    参数：
    version用于控制模型里面各个阶段特征图通道数的大小
    0.5：[48, 96, 192, 1024],
    1.0：[116, 232, 464, 1024]
    1.5：[176, 352, 704, 1024]
    2.0：[244, 488, 976, 2048]
    """
    def __init__(self,version,num_classes=10,dilated=False):
        super(ShuffleNetV2,self).__init__()
        
        self.stages_repeats=[3,7,3]#超参数，3个阶段的util重复次数
        if(version==0.5):
            self.stages_out_channels=[48, 96, 192, 1024]
        elif version == 1.0:
            self.stages_out_channels = [116, 232, 464, 1024]
        elif version == 1.5:
            self.stages_out_channels = [176, 352, 704, 1024]
        elif version == 2.0:
            self.stages_out_channels = [244, 488, 976, 2048]
        else:
            raise ValueError("Unknown version.")
        
        self.conv1=ConvBnRelu(3,24,3,2,1)
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.in_channels=24
        
        self.stage1=self.make_stage(self.stages_out_channels[0], self.stages_repeats[0],1)
        
        if dilated:#是否采用空洞卷积，是的话就dilation传入2(扩张2倍)
            self.stage2=self.make_stage(self.stages_out_channels[1], self.stages_repeats[1],2)
            self.stage3=self.make_stage(self.stages_out_channels[2], self.stages_repeats[2],2)
        else:
            self.stage2=self.make_stage(self.stages_out_channels[1], self.stages_repeats[1],1)
            self.stage3=self.make_stage(self.stages_out_channels[2], self.stages_repeats[2],1)
        self.conv5=ConvBnRelu(self.in_channels,self.stages_out_channels[-1],1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages_out_channels[-1], num_classes)
        
        
    def make_stage(self,out_channels,repeats,dilation=1):
        if(dilation==1):
            stride=2
        else:
            stride=1
            
        #每个阶段的第一个util才进行下采样
        layers=[ShuffleNetV2Util(self.in_channels, out_channels, stride, dilation)]
        
        self.in_channels=out_channels
        for i in range(repeats):
            #由于每个阶段的第一个util才进行下采样，所以其他的util的stride传入1
            layers.append(ShuffleNetV2Util(self.in_channels, out_channels, 1, 1))
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.conv5(x)
        x=self.avg_pool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

def get_shufflenet_v2(version,num_classes,dilated):
    model = ShuffleNetV2(version,num_classes,dilated)
    return model

if __name__ == '__main__':
    model=get_shufflenet_v2(0.5,10,False)
    print(model)
    summary(model, (3, 224, 224))    
