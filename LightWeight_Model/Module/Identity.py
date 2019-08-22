# -*- coding: utf-8 -*-
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self,in_channels):
        super(Identity,self).__init__()
    def forward(self,x):
        return x

