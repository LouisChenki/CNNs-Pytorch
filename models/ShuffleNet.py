import torch.nn as nn
import torch


class DWConv(nn.Module):
    def __init__(self, planes, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=planes)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GConv(nn.Module):
    def __init__(self, inplanes, planes, g):
        super(GConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, groups=g)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


'''
按照原先张量的数据组织方式(batchsize,channels, x, y)
无非就是channels的部分多加了一个维度，
然后将这两个维度转置后，
再合为一个维度
(b,c,h,w) -> (b,g,cpg,h,w) ->(b,cgp,g,h,w)->(b,c,h,w)
'''
def shuffle(x, num_group):
    batchsize, channels, height, width = x.size()
    # (b, c, h, w) -> (b, g, cpg, h, w)
    x = x.view(batchsize, num_group, channels // num_group, height, width)
    # (b,g,cpg,h,w) ->(b,cgp,g,h,w)
    x = torch.transpose(x, 1, 2).contiguous()
    # (b,cgp,g,h,w)->(b,c,h,w)
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, inplanes, planes, g, downsample=False, gf=False):
        super(ShuffleUnit, self).__init__()
        self.gconv1 = GConv(inplanes=inplanes, planes=planes // 4, g=g if not gf else 1)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv = DWConv(planes=planes // 4, stride=2 if downsample else 1)
        self.gconv2 = GConv(inplanes=planes // 4, planes=planes, g=g)
        self.shortcut = downsample
        self.group = g if not gf else 1

    def forward(self, x):
        if self.shortcut:
            identity = self.shortcut(x)
        else:
            identity = x
        x = self.gconv1(x)
        x = self.relu(x)
        x = shuffle(x, self.group)
        x = self.dwconv(x)
        x = self.gconv2(x)
        if self.shortcut:
            x = torch.cat((identity, x), dim=1)
        else:
            x += identity
        return x


class ShuffleNet(nn.Module):
    def __init__(self, g, x, num_class=1000):
        super(ShuffleNet, self).__init__()
        if g == 1:
            self.output_channels = [a * x for a in [24, 144, 288, 576]]
        elif g == 2:
            self.output_channels = [a * x for a in [24, 200, 400, 800]]
        elif g == 3:
            self.output_channels = [a * x for a in [24, 240, 480, 960]]
        elif g == 4:
            self.output_channels = [a * x for a in [24, 272, 544, 1088]]
        elif g == 8:
            self.output_channels = [a * x for a in [24, 384, 768, 1536]]
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = self._make_stage(inplanes=self.output_channels[0], planes=self.output_channels[1],  g=g, num=4)
        self.stage3 = self._make_stage(inplanes=self.output_channels[1], planes=self.output_channels[2], g=g, num=8)
        self.stage4 = self._make_stage(inplanes=self.output_channels[2], planes=self.output_channels[3], g=g, num=4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.output_channels[3], num_class)

    def forward(self, x):
        x = self.beginning(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_stage(self, inplanes, planes, g, num):
        stage = list()
        for i in range(num):
            # 是在每一个stage的第一个unit进行channels的翻倍
            if i != 0:
                stage.append(ShuffleUnit(inplanes=planes, planes=planes, g=g, downsample=False))
            else:
                if inplanes * 2 == planes:
                    downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                    stage.append(ShuffleUnit(inplanes=inplanes, planes=planes - inplanes, g=g, downsample=downsample))
                else:
                    downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                    stage.append(ShuffleUnit(inplanes=inplanes, planes=planes - inplanes, g=g, downsample=downsample, gf=True))
        return nn.Sequential(*stage)

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


def shufflenet_1x_g1(num_class=1000, initialize=False):
    net = ShuffleNet(g=1, x=1, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_1x_g2(num_class=1000, initialize=False):
    net = ShuffleNet(g=2, x=1, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_1x_g3(num_class=1000, initialize=False):
    net = ShuffleNet(g=3, x=1, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_1x_g4(num_class=1000, initialize=False):
    net = ShuffleNet(g=4, x=1, num_class=num_class)
    if initialize:
        net.initialization()
    return net


'''
def shufflenet_1x_g8(num_class=1000, initialize=False):
    net = ShuffleNet(g=8, x=1, num_class=num_class)
    if initialize:
        net.initialization()
    return net
'''


def shufflenet_05x_g1(num_class=1000, initialize=False):
    net = ShuffleNet(g=1, x=0.5, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_05x_g2(num_class=1000, initialize=False):
    net = ShuffleNet(g=2, x=0.5, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_05x_g3(num_class=1000, initialize=False):
    net = ShuffleNet(g=3, x=0.5, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_05x_g4(num_class=1000, initialize=False):
    net = ShuffleNet(g=4, x=0.5, num_class=num_class)
    if initialize:
        net.initialization()
    return net


'''
def shufflenet_05x_g8(num_class=1000, initialize=False):
    net = ShuffleNet(g=8, x=0.5, num_class=num_class)
    if initialize:
        net.initialization()
    return net
'''


def shufflenet_025x_g1(num_class=1000, initialize=False):
    net = ShuffleNet(g=1, x=0.25, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_025x_g2(num_class=1000, initialize=False):
    net = ShuffleNet(g=2, x=0.25, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_025x_g3(num_class=1000, initialize=False):
    net = ShuffleNet(g=3, x=0.25, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def shufflenet_025x_g4(num_class=1000, initialize=False):
    net = ShuffleNet(g=4, x=0.25, num_class=num_class)
    if initialize:
        net.initialization()
    return net


'''
def shufflenet_025x_g8(num_class=1000, initialize=False):
    net = ShuffleNet(g=8, x=0.25, num_class=num_class)
    if initialize:
        net.initialization()
    return net
'''

