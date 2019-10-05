import torch.nn as nn
import torch


# Figure 5 in paper
class Inception_A(nn.Module):
    def __init__(self, inplanes, planes):
        super(Inception_A, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=48, kernel_size=1, stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x11 = self.branch1x1(x)
        x33 = self.branch3x3(x)
        x55 = self.branch5x5(x)
        xpool = self.branchpool(x)
        return torch.cat((x11, x33, x55, xpool), 1)


# Figure 6 in paper
class Inception_B(nn.Module):
    def __init__(self, inplanes, planes):
        super(Inception_B, self).__init__()
        self.branch7x7t = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.branch7x7l = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=planes, out_channels=192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inplanes, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x77t = self.branch7x7t(x)
        x77l = self.branch7x7l(x)
        x11 = self.branch1x1(x)
        xpool = self.branchpool(x)
        return torch.cat((x77t, x77l, x11, xpool), 1)


# Figure 7 in paper
class Inception_C(nn.Module):
    def __init__(self, inplanes):
        super(Inception_C, self).__init__()
        self.sub_branch3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.sub_branch3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.branch3x3_left = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=448, kernel_size=1, stride=1),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=448, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            self.sub_branch3x3_1
        )
        self.branch3x3_right = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=448, kernel_size=1, stride=1),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=448, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            self.sub_branch3x3_2
        )
        self.sub_branch3x3l_1 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.sub_branch3x3l_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.branch3x3l_left = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=384, kernel_size=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            self.sub_branch3x3l_1
        )
        self.branch3x3l_right = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=384, kernel_size=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            self.sub_branch3x3l_2
        )
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=320, kernel_size=1, stride=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inplanes, out_channels=192, kernel_size=1)
        )

    def forward(self, x):
        x33 = torch.cat((self.branch3x3_left(x), self.branch3x3_right(x)), 1)
        x33l = torch.cat((self.branch3x3l_left(x), self.branch3x3l_right(x)), 1)
        x11 = self.branch1x1(x)
        xpool = self.branchpool(x)
        return torch.cat((x33, x33l, x11, xpool), 1)


class Inception_pool(nn.Module):
    def __init__(self, inplanes):
        super(Inception_pool, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.branch3x3l = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=384, kernel_size=3, stride=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x33 = self.branch3x3(x)
        x33l = self.branch3x3l(x)
        xpool = self.branchpool(x)
        return torch.cat((x33, x33l, xpool), 1)


class Inception_pool2(nn.Module):
    def __init__(self, inplanes):
        super(Inception_pool2, self).__init__()
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=320, kernel_size=3, stride=2),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=192, kernel_size=1, stride=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x33 = self.branch3x3(x)
        x77 = self.branch7x7(x)
        xpool = self.branchpool(x)
        return torch.cat((x33, x77, xpool), 1)


class Inception_V3(nn.Module):
    def __init__(self, num_class=1000):
        super(Inception_V3, self).__init__()
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        )
        self.i_5a = Inception_A(inplanes=192, planes=32)
        self.i_5b = Inception_A(inplanes=256, planes=64)
        self.i_5c = Inception_A(inplanes=288, planes=64)
        self.i_6a = Inception_pool(inplanes=288)
        self.i_6b = Inception_B(inplanes=768, planes=128)
        self.i_6c = Inception_B(inplanes=768, planes=160)
        self.i_6d = Inception_B(inplanes=768, planes=160)
        self.i_6e = Inception_B(inplanes=768, planes=192)
        self.i_7a = Inception_pool2(inplanes=768)
        self.i_7b = Inception_C(inplanes=1280)
        self.i_7c = Inception_C(inplanes=2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.beginning(x)
        x = self.i_5a(x)
        x = self.i_5b(x)
        x = self.i_5c(x)
        x = self.i_6a(x)
        x = self.i_6b(x)
        x = self.i_6c(x)
        x = self.i_6d(x)
        x = self.i_6e(x)
        x = self.i_7a(x)
        x = self.i_7b(x)
        x = self.i_7c(x)
        x = self.pool(x)
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


def inception_v3(num_class=1000, initialize=False):
    net = Inception_V3(num_class=num_class)
    if initialize:
        net.initialization()
    return net
