# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,vgg,num_classes=1000,init_weights=True,bn=False):
        super(VGG,self).__init__()
        if(vgg=='vgg11'):
            cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif(vgg=='vgg13'):
            cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        elif(vgg=='vgg16'):
            cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif(vgg=='vgg19'):
            cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        else:
            print("Incorrect parameters!")
        self.features=self.make_layers(cfg,bn)
        self.avgpool=nn.AdaptiveAvgPool2d((7, 7))
        self.classifier=nn.Sequential(
                nn.Linear(512*7*7,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096,num_classes)
                )
        if(init_weights):
            self._initialize_weights()
    
    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
    #构造vgg下采样的卷积块
    def make_layers(self,cfg,bn=False):
        layers=[]
        in_channels=3
        for v in cfg:
            if v=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                conv2d=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
                if(bn):
                    layers+=[conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                else:
                    layers+=[conv2d,nn.ReLU(inplace=True)]
                in_channels=v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



vgg11=VGG('vgg11',init_weights=True,bn=True)
vgg11








    



