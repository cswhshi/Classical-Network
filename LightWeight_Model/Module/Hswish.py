# -*- coding: utf-8 -*-
import torch.nn as nn 
"""
MobileNetV3中发现swish激活函数能够有效提高网络的精度，但是swish的计算量太大了，并不适合轻量级神经网络。
MobileNetV3找到了类似swish激活函数但是计算量却少很多的替代激活函数h-swish（hard version of swish）如下所示：
H_swish=x*relu6(x+3)/6
"""
class Hswish(nn.Module):
    def __init__(self,inplace=True):
        super(Hswish,self).__init__()
        self.relu6=nn.ReLU6(inplace)
    def forward(self,x):
        return x*self.relu6(x+3.)/6
