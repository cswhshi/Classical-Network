# -*- coding: utf-8 -*-
import torch.nn as nn
from LightWeight_Model.Module import Hswish,SEModule,Bottleneck,ConvBNswish


class MobileNetV3(nn.Module):
    def __init__(self, nclass=1000, mode='large', width_mult=1.0, dilated=False):
        super(MobileNetV3, self).__init__()
        if mode == 'large':
             # k, exp_size, c, se, nl, s
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1], ]
            layer2_setting = [
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1], ]
            layer3_setting = [
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1], ]
            layer4_setting = [
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1], ]
        elif mode == 'small':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, 'RE', 2], ]
            layer2_setting = [
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1], ]
            layer3_setting = [
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1], ]
            layer4_setting = [
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1], ]
        else:
            raise ValueError('Unknown mode.')

        # building first layer
        self.in_channels = int(16 * width_mult)
        self.conv1 = ConvBNswish(3, self.in_channels, 3, 2, 1)
        
        # building bottleneck blocks
        self.layer1 = self.make_layer(Bottleneck, layer1_setting,width_mult)
        self.layer2 = self.make_layer(Bottleneck, layer2_setting,width_mult)
        self.layer3 = self.make_layer(Bottleneck, layer3_setting,width_mult)
        if dilated:
            self.layer4 = self.make_layer(Bottleneck, layer4_setting,width_mult, dilation=2)
        else:
            self.layer4 = self.make_layer(Bottleneck, layer4_setting,width_mult)
            
        # building last several layers
        classifier = list()
        if mode == 'large':
            last_bneck_channels = int(960 * width_mult) if width_mult > 1.0 else 960
            self.layer5 = ConvBNswish(self.in_channels, last_bneck_channels, 1)
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        elif mode == 'small':
            last_bneck_channels = int(576 * width_mult) if width_mult > 1.0 else 576
            self.layer5 = ConvBNswish(self.in_channels, last_bneck_channels, 1)
            classifier.append(SEModule(last_bneck_channels))
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        else:
            raise ValueError('Unknown mode.')
        self.classifier = nn.Sequential(*classifier)
        self.init_weights()

    def make_layer(self, block, block_setting, width_mult, dilation=1):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            exp_channels = int(exp_size * width_mult)
            layers.append(block(self.in_channels, out_channels, exp_channels, k, stride, dilation, se, nl))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def init_weights(self):
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
        return x
