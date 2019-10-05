import torch.nn as nn


class Dsconv(nn.Module):
    ex = 2

    def __init__(self, inplanes, planes, s):
        super(Dsconv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=inplanes, out_channels=inplanes, kernel_size=3, stride=s, padding=1, groups=inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, num_class=1000):
        super(MobileNet, self).__init__()
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = Dsconv(inplanes=32, planes=64, s=1)
        self.layer2 = Dsconv(inplanes=64, planes=128, s=2)
        self.layer3 = Dsconv(inplanes=128, planes=128, s=1)
        self.layer4 = Dsconv(inplanes=128, planes=256, s=2)
        self.layer5 = Dsconv(inplanes=256, planes=256, s=1)
        self.layer6 = Dsconv(inplanes=256, planes=512, s=2)
        self.layer7 = self._make_layer(Dsconv, 5, 512)
        self.layer8 = Dsconv(inplanes=512, planes=1024, s=2)
        self.layer9 = Dsconv(inplanes=1024, planes=1024, s=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_class)

    def _make_layer(self, conv, num, inplanes):
        layer = list()
        for i in range(num):
            layer.append(conv(inplanes=inplanes, planes=inplanes, s=1))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.beginning(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
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
            elif isinstance(per, nn.BatchNorm2d):
                nn.init.constant_(per.weight, 1)
                nn.init.constant_(per.bias, 0)


def mobilenet_v1(num_class=1000, initialize=False):
    net = MobileNet(num_class=num_class)
    if initialize:
        net.initialization()
    return net
