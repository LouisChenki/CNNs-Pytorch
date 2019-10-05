import torch.nn as nn
import torch


class Inception(nn.Module):
    def __init__(self, inplanes, x11, x33r, x33, x55r, x55, proj):
        super(Inception, self).__init__()
        self.flow1 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=x11, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.flow2 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=x33r, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=x33r, out_channels=x33, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flow3 = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=x55r, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=x55r, out_channels=x55, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )
        self.flow4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inplanes, out_channels=proj, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.flow1(x)
        f2 = self.flow2(x)
        f3 = self.flow3(x)
        f4 = self.flow4(x)
        return torch.cat((f1, f2, f3, f4), 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_class=1000):
        super(GoogLeNet, self).__init__()
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.i_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.i_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.i_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.i_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.i_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.i_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.i_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.i_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.i_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )
        self.fc = nn.Linear(1000, num_class)

    def forward(self, x):
        x = self.beginning(x)
        x = self.i_3a(x)
        x = self.i_3b(x)
        x = self.pool1(x)
        x = self.i_4a(x)
        x = self.i_4b(x)
        x = self.i_4c(x)
        x = self.i_4d(x)
        x = self.i_4e(x)
        x = self.pool2(x)
        x = self.i_5a(x)
        x = self.i_5b(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        x = self.fc(x)
        return x

    def initialization(self):
        for per in self.modules():
            if isinstance(per, nn.Conv2d):
                nn.init.kaiming_normal_(per.weight, per.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(per, nn.Linear):
                nn.init.normal_(per.weight, 0, 0.01)
                nn.init.constant_(per.bias, 0)


def googlenet(num_class=1000, initialize=False):
    net = GoogLeNet(num_class=num_class)
    if initialize:
        net.initialization()
    return net