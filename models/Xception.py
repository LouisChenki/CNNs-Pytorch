import torch
import torch.nn as nn


class SeparableConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=3, stride=1, padding=1, groups=inplanes)
        self.pointwise = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, outplanes, shortcut=None, first=None, downsample=None):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.sepconv1 = SeparableConv(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sepconv2 = SeparableConv(planes, outplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shortcut = shortcut
        self.first = first
        self.downsample = downsample
        self.downsample_shortcut = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=2, padding=0)
        self.downsample_flow = nn.Sequential(
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            SeparableConv(outplanes, outplanes)
        )

    def forward(self, x):
        if self.shortcut is False:
            identity = x
        else:
            identity = self.downsample_shortcut(x)
        if self.first is False:
            x = self.bn1(x)
            x = self.relu(x)
        x = self.sepconv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.sepconv2(x)
        if self.downsample is True:
            x = self.maxpool(x)
        else:
            x = self.downsample_flow(x)
        x += identity
        return x


class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = Block(inplanes=64, planes=128, outplanes=128, shortcut=True, first=True, downsample=True)
        self.block2 = Block(inplanes=128, planes=256, outplanes=256, shortcut=True, first=False, downsample=True)
        self.block3 = Block(inplanes=256, planes=728, outplanes=728, shortcut=True, first=False, downsample=True)

    def forward(self, x):
        x = self.beginning(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class MiddleFlow(nn.Module):
    def __init__(self):
        super(MiddleFlow, self).__init__()
        self.block = self._make_layer(Block, 8)

    def _make_layer(self, block, num):
        layer = list()
        for i in range(num):
            layer.append(Block(inplanes=728, planes=728, outplanes=728, shortcut=False, first=False, downsample=False))
        return nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x)


class ExitFlow(nn.Module):
    def __init__(self, num_class=1000):
        super(ExitFlow, self).__init__()
        self.block = Block(inplanes=728, planes=728, outplanes=1024, shortcut=True, first=False, downsample=True)
        self.end = nn.Sequential(
            SeparableConv(inplanes=1024, planes=1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv(inplanes=1536, planes=2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(in_features=2048, out_features=num_class)

    def forward(self, x):
        x = self.block(x)
        x = self.end(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# input 299x299
class Xception(nn.Module):
    def __init__(self, num_class=1000):
        super(Xception, self).__init__()
        self.entryflow = EntryFlow()
        self.middleflow = MiddleFlow()
        self.exitflow = ExitFlow(num_class=num_class)

    def forward(self, x):
        x = self.entryflow(x)
        x = self.middleflow(x)
        x = self.exitflow(x)
        return x

    def initialization(self):
        for per in self.modules():
            if isinstance(per, nn.Conv2d):
                nn.init.kaiming_normal_(per.weight, per.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(per, nn.Linear):
                nn.init.normal_(per.weight, 0, 0.01)
                nn.init.constant_(per.bias, 0)
            elif isinstance(per, nn.BatchNorm2d):
                nn.init.constant_(per.weight, 1)
                nn.init.constant_(per.bias, 0)


def xception(num_class=1000, initialize=False):
    net = Xception(num_class=num_class)
    if initialize:
        net.initialization()
    return net

