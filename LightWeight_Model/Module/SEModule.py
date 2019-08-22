# -*- coding: utf-8 -*-
import torch.nn as nn
from LightWeight_Model.Module.Hsigmoid import Hsigmoid
from torchsummary import summary
"""
SE模块:
    1、MobileNetV3的SE模块被运用在线性瓶颈结构最后一层上，代替V2中最后的逐点卷积，改为先进行SE操作再逐点卷积。
      这样保持了网络结构每层的输入和输出，仅在中间做处理。
    2、SE模块的细节：
        （1）、压缩，使用全局平均池化（global average pooling），操作后特征图被压缩为1×1×C向量。
        （2）、激励（Excitation）操作，由两个全连接层组成，其中SERatio是一个缩放参数，
                这个参数的目的是为了减少通道个数从而降低计算量。
                第一个全连接层有C*SERatio个神经元，输入为1×1×C，输出1×1×C×SERadio。
                第二个全连接层有C个神经元，输入为1×1×C×SERadio，输出为1×1×C。
        （3）、scale操作，在得到1×1×C向量之后，就可以对原来的特征图进行scale操作了。很简单，
                就是通道权重相乘，原有特征向量为W×H×C，将SE模块计算出来的各通道权重值分别
                和原特征图对应通道的二维矩阵相乘，得出的结果输出。
"""
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
    
if __name__ == "__main__":
    SE=SEModule(in_channels=12)
    print(SE)
    summary(SE, (12, 224, 224))
 





