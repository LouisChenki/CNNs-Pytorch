import torch.nn as nn



def Conv3x3(in_f, out_f):
    return nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=3, stride=1, padding=1)


def Conv1x1(in_f, out_f):
    return nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=1, stride=1)


def MaxPool():
    return nn.MaxPool2d(kernel_size=2, stride=2)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, num, lrn=False, smallconv=False):
        super(BasicBlock, self).__init__()
        self.block = self.make_layer(inplanes, planes, num, lrn, smallconv)

    def make_layer(self, inplanes, planes, num, lrn=False, smallconv=False):
        block = list()
        block.append(Conv3x3(in_f=inplanes, out_f=planes))
        block.append(nn.ReLU(inplace=True))
        for i in range(1, num-1):
            block.append(Conv3x3(in_f=planes, out_f=planes))
            block.append(nn.ReLU(inplace=True))
        if lrn is True:
            block.append(nn.LocalResponseNorm(size=5))
        elif smallconv is True:
            block.append(Conv1x1(in_f=planes, out_f=planes))
            block.append(nn.ReLU(inplace=True))
        elif num > 1:
            block.append(Conv3x3(in_f=planes, out_f=planes))
            block.append(nn.ReLU(inplace=True))
        return nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)
        return x


class Classifer(nn.Module):
    def __init__(self, num_class=1000):
        super(Classifer, self).__init__()
        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_class)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return x


class VGG(nn.Module):
    def __init__(self, num, lrn=False, smallconv=False, num_class=1000):
        super(VGG, self).__init__()
        self.block1 = BasicBlock(inplanes=3, planes=64, num=num[0], lrn=lrn, smallconv=False)
        self.block2 = BasicBlock(inplanes=64, planes=128, num=num[1])
        self.block3 = BasicBlock(inplanes=128, planes=256, num=num[2], smallconv=smallconv)
        self.block4 = BasicBlock(inplanes=256, planes=512, num=num[3], smallconv=smallconv)
        self.block5 = BasicBlock(inplanes=512, planes=512, num=num[4], smallconv=smallconv)
        self.pool = MaxPool()
        self.classifer = Classifer(num_class=num_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.block5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x

    def initialization(self):
        for per in self.modules():
            if isinstance(per, nn.Conv2d):
                nn.init.kaiming_normal_(per.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(per, nn.Linear):
                nn.init.normal_(per.weight, 0, 0.01)
                nn.init.constant_(per.bias, 0)


def vgg_11(num_class, initialize=False):
    net = VGG(num=[1, 1, 2, 2, 2], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def vgg_11_lrn(num_class, initialize=False):
    net = VGG(num=[1, 1, 2, 2, 2], lrn=True, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def vgg_13(num_class, initialize=False):
    net = VGG(num=[2, 2, 2, 2, 2], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def vgg_16_c(num_class, initialize=False):
    net = VGG(num=[2, 2, 3, 3, 3], smallconv=True, num_class=num_class)
    if initialize:
        net.initialization()
    return net


def vgg_16_d(num_class, initialize=False):
    net = VGG(num=[2, 2, 3, 3, 3], num_class=num_class)
    if initialize:
        net.initialization()
    return net


def vgg_19(num_class, initialize=False):
    net = VGG(num=[2, 2, 4, 4, 4], num_class=num_class)
    if initialize:
        net.initialization()
    return net