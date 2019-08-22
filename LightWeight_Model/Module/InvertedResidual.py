# -*- coding: utf-8 -*-
import torch.nn as nn
from torchsummary import summary
from LightWeight_Model.Module.ConvBnRelu import ConvBnRelu


"""
MobileNetV2:
    MobileNetV2版本中的残差结构使用第一层逐点卷积升维并使用Relu6激活函数代替Relu，
    之后使用深度卷积，同样使用Relu6激活函数，再使用逐点卷积降维，降维后使用Linear激活函数。
    这样的卷积操作方式更有利于移动端使用（有利于减少参数与M-Adds计算量），因维度升降方式
    与ResNet中的残差结构刚好相反，MobileNetV2将其称之为反向残差（Inverted Residuals）。
"""
class InvertedResidual(nn.Module):
    def __init__(self,in_channels,out_channels,stride,expand_ratio,dilation=1):
        super(InvertedResidual,self).__init__()
        '''
        assert 断言语句和 if 分支有点类似，它用于对一个 bool 表达式
        进行断言，如果该 bool 表达式为 True，该程序可以继续向下执行；
        否则程序会引发 AssertionError 错误。
        '''
        assert stride in [1,2]
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        
        layers=[]
        inter_channels=int(round(in_channels * expand_ratio))#round：四舍五入
        if(expand_ratio!=1):
            layers.append(ConvBnRelu(in_channels,inter_channels,1,relu6=True))
        '''
        padding=dilation是因为如果使用dilation>1的话，padding也是跟着要扩大的，这样才能保证输入尺寸和输出尺寸一样
        '''
        #dw
        layers.append(ConvBnRelu(inter_channels,inter_channels,3,stride,padding=dilation,dilation=dilation,groups=inter_channels, relu6=True))
        layers.append(nn.Conv2d(inter_channels,out_channels,1,bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv=nn.Sequential(*list(layers))
    def forward(self,x):
        '''
        线性瓶颈结构有两种:
            第一种是步长为1时使用残差结构，
            第二种是步长为2时不使用残差结构
        '''
        if(self.use_res_connect):
            return x+self.conv(x)
        else:
            return self.conv(x)

    
if __name__ == "__main__":
    IRes=InvertedResidual(in_channels=3,out_channels=3,stride=1,expand_ratio=3,dilation=1)
    print(IRes)
    summary(IRes,(3, 224, 224))    
    
