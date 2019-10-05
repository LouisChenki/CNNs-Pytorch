import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, inplanes, r=16):
        super(Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channels=inplanes, out_channels=inplanes // r, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=inplanes // r, out_channels=inplanes, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sig(out)
        return x * out


class Bottleneck(nn.Module):
    ex = 4

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.ex, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes * self.ex)
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
        self.attention = Attention(inplanes=planes * self.ex)

    def forward(self, x):
        if self.shortcut is None:
            identity = x
        else:
            identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.attention(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_class=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = self._make_layer(block=block, num=block_num[0], planes=64, stride=1)
        self.conv3 = self._make_layer(block=block, num=block_num[1], planes=128, stride=2)
        self.conv4 = self._make_layer(block=block, num=block_num[2], planes=256, stride=2)
        self.conv5 = self._make_layer(block=block, num=block_num[3], planes=512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_class)

    def _make_layer(self, block, num, planes, stride):
        if stride > 1 or self.inplanes != planes * block.ex:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.ex, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.ex)
            )
        else:
            shortcut = None
        layer = list()
        layer.append(block(self.inplanes, planes, stride=stride, shortcut=shortcut))
        self.inplanes = planes * block.ex
        for i in range(1, num):
            layer.append(block(self.inplanes, planes, stride=1, shortcut=None))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
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


def se_resnet50(num_class=1000, initialize=False):
    net = ResNet(Bottleneck, block_num=[3, 4, 6, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def se_resnet101(num_class=1000, initialize=False):
    net = ResNet(Bottleneck, block_num=[3, 4, 23, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def se_resnet152(num_class=1000, initialize=False):
    net = ResNet(Bottleneck, block_num=[3, 8, 36, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net