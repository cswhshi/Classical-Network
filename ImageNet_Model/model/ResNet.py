# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1):
        super(BasicBlock, self).__init__()
        self.residual=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels*BasicBlock.expansion,kernel_size=3,padding=1,bias=False),
                nn.BatchNorm2d(out_channels*BasicBlock.expansion)
                )
        self.shortcut=nn.Sequential()
        
        #如果stride不为1或者输出特征图的维度不等于输入特征图的维度,需要对输入进行处理之后再进行短接
        if stride!=1 or in_channels!=out_channels*BasicBlock.expansion:
            self.shortcut=nn.Sequential(
                    nn.Conv2d(in_channels,out_channels*BasicBlock.expansion,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(out_channels*BasicBlock.expansion)
                    )
    def forward(self,x):
        out=self.residual(x)
        x=self.shortcut(x)
        return nn.ReLU(out+x)
    
class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    与上面的block不同，这个block采用了1*1的结构，目的主要是为了降低参数的数目，
    假设输入特征图的为256个通道，expansion=4，那么第一个1x1的卷积把256维channel降到64维，然后在最后通过1x1卷积恢复维度，
    整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632，
    而不使用bottleneck的话就是两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648，差了16.94倍。 
    """
    expansion=4
    def __init__(self,in_channels,out_channels,stride=1):
        super(BottleNeck,self).__init__()
        self.residual=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))
        self.shortcut==nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
    def forward(self,x):
        out=self.residual(x)
        x=self.shortcut(x)
        return nn.ReLU(out+x)
    
class ResNet(nn.Module):
    def __init__(self,block,num_block,num_classes=1000):
        super(ResNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self,block,out_channels,num_blocks, stride):
        """
        block:选用的build block
        out_channels:输出特征图的通道数
        num_block:block的个数
        stride:步幅，仅仅作用有第一个相同分辨率的特征图的第一个block
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 
    
    
def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])  
def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

resnet18()

    
    