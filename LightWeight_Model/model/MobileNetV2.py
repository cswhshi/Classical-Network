# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from LightWeight_Model.Module import ConvBnRelu,InvertedResidual

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=10,width_mult=1.0,dilated=False):
        super(MobileNetV2,self).__init__()
        # t, c, n, s
        #层的设置，t,c, n, s，分别表示的是expand_ratio,输出通道数，堆叠数，stride，其中stride大于1将会使分辨率降低
        layer1_setting = [
            [1, 16, 1, 1]]
        layer2_setting = [
            [6, 24, 2, 2]]
        layer3_setting = [
            [6, 32, 3, 2]]
        layer4_setting = [
            [6, 64, 4, 2],
            [6, 96, 3, 1]]
        layer5_setting = [
            [6, 160, 3, 2],
            [6, 320, 1, 1]]
        
        #building first layer
        self.in_channels = int(32 * width_mult)
        last_channels = int(1280 * width_mult)
        self.conv1=ConvBnRelu(3,self.in_channels,3,2,1,relu6=True)##分辨率降低到原来的一半
        
        # building inverted residual blocks
        self.layer1 = self._make_layer(InvertedResidual, layer1_setting, width_mult)
        self.layer2 = self._make_layer(InvertedResidual, layer2_setting, width_mult)
        self.layer3 = self._make_layer(InvertedResidual, layer3_setting, width_mult)
        if dilated:
            self.layer4 = self._make_layer(InvertedResidual, layer4_setting, width_mult,
                                           dilation=2)
            self.layer5 = self._make_layer(InvertedResidual, layer5_setting, width_mult,
                                           dilation=2)
        else:
            self.layer4 = self._make_layer(InvertedResidual, layer4_setting, width_mult)
            self.layer5 = self._make_layer(InvertedResidual, layer5_setting, width_mult)
        
        # building last several layers
        self.classifier=nn.Sequential(
                ConvBnRelu(self.in_channels,last_channels,1,relu6=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Dropout(0.2),
                nn.Conv2d(last_channels,num_classes,1))
        self._init_weight()
    def _make_layer(self,block,block_setting,width_mult,dilation=1):
        layers=list()
        for t,c,n,s in block_setting:
            out_channels=int(c*width_mult)
            stride = s if (dilation == 1) else 1
            layers.append(block(self.in_channels,out_channels,stride,t,dilation))
            self.in_channels=out_channels
            for i in range(n-1):
                layers.append(block(self.in_channels,out_channels,1,t,1,))
                self.in_channels=out_channels
        return nn.Sequential(*layers)
    
    def _init_weight(self):
        # weight initialization
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
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x
    
def mobilenetV2(num_classes=10,width_mult=1.0,dilated=False):
    model= MobileNetV2(num_classes=num_classes,width_mult=width_mult,dilated=dilated)
    return model   

input = torch.rand(1,3,224,224)
model=mobilenetV2(width_mult=1.0)
print(model)

from torchsummary import summary
summary(model, (3, 224, 224))