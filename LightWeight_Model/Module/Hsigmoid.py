# -*- coding: utf-8 -*-
import torch.nn as nn
class Hsigmoid(nn.Module):
    def __init__(self,inplace=True):
        super(Hsigmoid,self).__init__()
        self.relu6=nn.ReLU6(inplace)
    def forward(self,x):
        return self.relu6(x+3.)/6.
