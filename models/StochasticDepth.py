import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    ex = 1

    def __init__(self, inplanes, planes, p, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
        self.probability = p

    def forward(self, x):
        # 训练的时候
        if self.training:
            survival = torch.rand(1)
            # 存活的情况
            if survival <= self.probability:
                if self.shortcut is None:
                    identity = x
                else:
                    identity = self.shortcut(x)
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x += identity
                x = self.relu(x)
            # 不存活的情况
            else:
                if self.shortcut is not None:
                    x = self.shortcut(x)
                    x = self.relu(x)
        # 测试的时候
        else:
            if self.shortcut is None:
                identity = x
            else:
                identity = self.shortcut(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = x * self.probability
            x += identity
            x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    ex = 4

    def __init__(self, inplanes, planes, p, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes * self.ex, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes * self.ex)
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)
        self.probability = p

    def forward(self, x):
        # 训练的情况
        if self.training:
            survival = torch.rand(1)
            # 存活的情况
            if survival <= self.probability:
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
                x += identity
                x = self.relu(x)
                # 不存活的情况
            else:
                if self.shortcut is not None:
                    x = self.shortcut(x)
                    x = self.relu(x)
        # 测试的情况
        else:
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
            x = x * self.probability
            x += identity
            x = self.relu(x)
        return x


class StochasticDepth(nn.Module):
    def __init__(self, block, block_num, pL=0.5, num_class=1000):
        super(StochasticDepth, self).__init__()
        self.l = 0
        self.pL = pL
        # 每一层的生存概率
        self.pl = [(1 - ((i / sum(block_num) * (1 - self.pL)))) for i in range(sum(block_num))]

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
        layer.append(block(self.inplanes, planes, stride=stride, shortcut=shortcut, p=self.pl[self.l]))
        self.inplanes = planes * block.ex
        self.l += 1
        for i in range(1, num):
            layer.append(block(self.inplanes, planes, stride=1, shortcut=None, p=self.pl[self.l]))
            self.l += 1
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


def stochastic_depth18(num_class=1000, initialize=False):
    net = StochasticDepth(BasicBlock, block_num=[2, 2, 2, 2], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def stochastic_depth34(num_class=1000, initialize=False):
    net = StochasticDepth(BasicBlock, block_num=[3, 4, 6, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def stochastic_depth50(num_class=1000, initialize=False):
    net = StochasticDepth(Bottleneck, block_num=[3, 4, 6, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def stochastic_depth101(num_class=1000, initialize=False):
    net = StochasticDepth(Bottleneck, block_num=[3, 4, 23, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def stochastic_depth152(num_class=1000, initialize=False):
    net = StochasticDepth(Bottleneck, block_num=[3, 8, 36, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net