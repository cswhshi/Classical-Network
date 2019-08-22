# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

__all__=['ConvBnRelu','DWConvBnRelu','InvertedResidual']#,'_ConvBNReLU','_DWConvBNReLU']

"""
Conv-Bn-Relu moduule
"""
class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,relu6=False,**kwargs):
        super(ConvBnRelu,self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True) 
        
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x


"""
convBR=ConvBnRelu(in_channels=3,out_channels=32,kernel_size=3)
convBR
from torchsummary import summary
summary(convBR, (3, 224, 224))



convBR=ConvBnRelu(in_channels=5,out_channels=5,kernel_size=3)
convBR

Out[1]: 
ConvBnRelu(
  (conv): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), bias=False)
  (bn): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace)
)
"""



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

"""
dwcbr=DWConvBnRelu(3,6,1,1)
dwcbr

Out[14]: 
DWConvBnRelu(
  (conv): Sequential(
    (0): ConvBnRelu(
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), groups=3, bias=False)
      (bn): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
    (1): ConvBnRelu(
      (conv): Conv2d(3, 6, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
    )
  )
)
    
"""


"""
MobileNetV2:InvertedResidual
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
"""   
IRes=InvertedResidual(6,6,1,3)
IRes

"""

# -----------------------------------------------------------------
#                      For MobileNetV3
# -----------------------------------------------------------------

class Hswish(nn.Module):
    def __init__(self,inplace=True):
        super(Hswish,self).__init__()
        self.relu6=nn.ReLU6(inplace)
    def forward(self,x):
        return x*self.relu6(x+3.)/6

class Hsigmoid(nn.Module):
    def __init__(self,inplace=True):
        super(Hsigmoid,self).__init__()
        self.relu6=nn.ReLU6(inplace)
    def forward(self,x):
        return self.relu6(x+3.)/6.

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

class SEModule(nn.Module):
    '''
    reduction参数：是一个缩放参数，这个参数的目的是为了减少通道个数从而降低计算量。
    
    '''
    def __init__(self,in_channels,reduction=4):
        super(SEModule,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
                nn.Linear(in_channels,in_channels//reduction,bias=False),
                nn.ReLU(True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                Hsigmoid(True))
    def forward(self,x):
        n,c,_,_=x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out.expand_as(x)



















