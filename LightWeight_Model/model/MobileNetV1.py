# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from LightWeight_Model.Module import ConvBnRelu,DWConvBnRelu

class MobileNet(nn.Module):
    def __init__(self,num_classes=10,width_mult=1.0,dilated=False):
        super(MobileNet,self).__init__()
        
        """
        层的设置，c, n, s，分别表示的是输出通道数，堆叠数，stride，其中stride大于1将会使分辨率降低
        """
        layer1_setting = [
            [64, 1, 1]]
        layer2_setting = [
            [128, 2, 2]]
        layer3_setting = [
            [256, 2, 2]]
        layer4_setting = [
            [512, 6, 2]]
        layer5_setting = [
            [1024, 2, 2]]
        
        """
        width_mult为宽度因子
        
        """
        self.in_channels=int(32*width_mult)
        
        """
        分辨率计算
        输入图片大小 W×W
        Filter大小 F×F
        步长 S
        padding的像素数 P
        于是我们可以得出
        N = (W − F + 2P )/S+1
        """
        self.conv1=ConvBnRelu(3,self.in_channels,3,2,1)   #分辨率降低到原来的一半
        
        """
        构建重复堆叠的一些层
        """
        self.layer1 = self.make_layer(DWConvBnRelu, layer1_setting, width_mult)
        self.layer2 = self.make_layer(DWConvBnRelu, layer2_setting, width_mult)
        self.layer3 = self.make_layer(DWConvBnRelu, layer3_setting, width_mult)
        
        #是否使用空洞卷积扩大感受野
        """
        空洞卷积的目的是为了在扩大感受野的同时，不降低图片分辨率和不引入额外参数及计算量
        （一般在CNN中扩大感受野都需要使用s>1的conv或者pooling，导致分辨率降低，
        不利于segmentation。如果使用大卷积核，确实可以达到增大感受野，但是会引入额外的参数及计算量）
        """
        if dilated:
            self.layer4 = self.make_layer(DWConvBnRelu, layer4_setting, width_mult,dilation=2)
            self.layer5 = self.make_layer(DWConvBnRelu, layer5_setting, width_mult,dilation=2)
        else:
            self.layer4 = self.make_layer(DWConvBnRelu, layer4_setting, width_mult)
            self.layer5 = self.make_layer(DWConvBnRelu, layer5_setting, width_mult)
            
        #分类网络结构，采用自适应平均池化代替全连接结构
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(1024 * width_mult), num_classes, 1))
        
        #初始化网络参数
        self._init_weights()
    """
    不同分辨率的特征图都会重复堆叠DWConvBnRelu，make_layer函数根据参数构建这些堆叠块
    """
    def make_layer(self,block,block_setting,width_mult,dilation=1):
        layers=list()
        for c,n,s in block_setting:
            out_channels=int(c*width_mult)
            stride = s if (dilation == 1) else 1#如果使用空洞卷积的话，也就是dilation>1，分辨率就不进行缩小了
            layers.append(
                block(self.in_channels, out_channels, stride, dilation))
            self.in_channels = out_channels
            for i in range(n - 1):
                layers.append(block(self.in_channels,out_channels, 1, 1))
                self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _init_weights(self):
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
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        x = x.view(x.size(0), x.size(1))
        #"""
        return x
        
def MobileNetV1(num_classes=10,width_mult=1.0,dilated=False):
    model= MobileNet(num_classes=num_classes,width_mult=width_mult,dilated=dilated)
    return model              

def mobilenet1_0(**kwargs):
    return MobileNetV1(width_mult=1.0, **kwargs)

def mobilenet0_75(**kwargs):
    return MobileNetV1(width_mult=0.75, **kwargs) #get_mobilenet(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    return MobileNetV1(width_mult=0.5, **kwargs)


def mobilenet0_25(**kwargs):
    return MobileNetV1(width_mult=0.2, **kwargs)


input = torch.rand(1,3,224,224)
model=mobilenet1_0()
print(model)
from torchsummary import summary
summary(model, (3, 224, 224))












