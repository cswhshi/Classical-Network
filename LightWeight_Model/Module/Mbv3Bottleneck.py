# -*- coding: utf-8 -*-
import torch.nn as nn
from LightWeight_Model.Module.Hsigmoid import Hsigmoid
from LightWeight_Model.Module.Hswish import Hswish
from LightWeight_Model.Module.SEModule import SEModule
from LightWeight_Model.Module.Identity import Identity


"""
MobileNetV3的Block
三个必要步骤：
    1×1卷积，由输入通道，转换为膨胀通道；
    3×3或5×5卷积，膨胀通道，使用步长stride；
    1×1卷积，由膨胀通道，转换为输出通道。
两个可选步骤：
    SE结构：Squeeze-and-Excite；
    连接操作，Residual残差；步长为1，同时输入和输出通道相同；
    其中激活函数有两种：ReLU和h-swish。

"""
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,exp_size,kernel_size,
                 stride,dilation=1,se=False,nl='RE'):
        super(Bottleneck,self).__init__()
        assert stride in [1,2]  
        #当stride为1且输入和输出通道一样时，才使用resnet的连接方式
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        
        """
        选择使用的激活函数类型
        """
        if nl=='HS':
            act=Hswish
        else:
            act=nn.ReLU
        """
        是否使用SE,不使用的话就什么都不做（Identity）
        """
        if se:
            SELayer=SEModule
        else:
            SELayer=Identity       
        self.conv=nn.Sequential(
                #pw
                nn.Conv2d(in_channels,exp_size,1,bias=False),
                nn.BatchNorm2d(exp_size),
                act(True),
                #dw
                nn.Conv2d(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                          dilation, groups=exp_size, bias=False),
                nn.BatchNorm2d(exp_size),
                SELayer(exp_size),
                act(True),
                #pw-linear
                nn.Conv2d(exp_size,out_channels,1,bias=False),
                nn.BatchNorm2d(out_channels))
    def forward(self,x):
        if self.use_res_connect:
            return x+self.conv(x)
        else:
            return self.conv(x)