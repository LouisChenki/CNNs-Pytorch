import torch.nn as nn
import torch


class Squeeze(nn.Module):
    def __init__(self, inplanes, planes):
        super(Squeeze, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Expand(nn.Module):
    def __init__(self, inplanes, planes1, planes3):
        super(Expand, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=inplanes, out_channels=planes3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x = self.conv2(x)
        x = self.relu(x)
        return torch.cat((x1, x), 1)


class Fire(nn.Module):
    def __init__(self, inplanes, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()
        self.squeeze = Squeeze(inplanes=inplanes, planes=s1x1)
        self.expand = Expand(inplanes=s1x1, planes1=e1x1, planes3=e3x3)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.expand(x)
        return x


class SqueezeNet(nn.Module):
    def __init__(self, num_class=1000):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire2 = Fire(inplanes=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire3 = Fire(inplanes=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire4 = Fire(inplanes=128, s1x1=32, e1x1=128, e3x3=128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire5 = Fire(inplanes=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire6 = Fire(inplanes=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire7 = Fire(inplanes=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire8 = Fire(inplanes=384, s1x1=64, e1x1=256, e3x3=256)
        self.pool8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fire9 = Fire(inplanes=512, s1x1=64, e1x1=256, e3x3=256)
        # 这里为了便于分类相对与论文原文有所改动
        self.pool9 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.pool4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.pool8(x)
        x = self.fire9(x)
        x = self.pool9(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def initialization(self):
        for per in self.modules():
            if isinstance(per, nn.Conv2d):
                nn.init.kaiming_normal_(per.weight, per.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(per, nn.Linear):
                nn.init.normal_(per.weight, 0, 0.01)
                nn.init.constant_(per.bias, 0)


def squeezenet(num_class=1000, initialize=False):
    net = SqueezeNet(num_class=num_class)
    if initialize:
        net.initialization()
    return net