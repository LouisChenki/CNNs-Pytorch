import torch.nn as nn


# 在预激活模块的基础上加入dropout
class BasicBlock(nn.Module):
    ex = 1

    def __init__(self, inplanes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        if self.shortcut is None:
            identity = x
        else:
            identity = self.shortcut(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += identity
        return x


class Wide_ResNet(nn.Module):
    def __init__(self, block, depth, wide, num_class=1000):
        super(Wide_ResNet, self).__init__()
        net_planes = [16 * wide, 32 * wide, 64 * wide]
        net_depth = int(((depth - 4) / 2) / 3)
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = self._make_layer(block=block, num=net_depth, planes=net_planes[0], stride=1)
        self.conv3 = self._make_layer(block=block, num=net_depth, planes=net_planes[1], stride=2)
        self.conv4 = self._make_layer(block=block, num=net_depth, planes=net_planes[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_class)

    def _make_layer(self, block, num, planes, stride):
        if stride > 1 or self.inplanes != planes * block.ex:
            shortcut = nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.ex, kernel_size=1, stride=stride)
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


def wide_resnet40_4(num_class=1000, initialize=False):
    net = Wide_ResNet(BasicBlock, depth=40, wide=4, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def wide_resnet16_8(num_class=1000, initialize=False):
    net = Wide_ResNet(BasicBlock, depth=16, wide=8, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def wide_resnet28_10(num_class=1000, initialize=False):
    net = Wide_ResNet(BasicBlock, depth=28, wide=10, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def resnet110(num_class=1000, initialize=False):
    net = Wide_ResNet(BasicBlock, depth=110, wide=1, num_class=num_class)
    if initialize:
        net.initialization()
    return net