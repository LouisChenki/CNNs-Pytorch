import torch.nn as nn
import torch


class Conv(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):
    ex = 2

    def __init__(self, inplanes, planes, f, c):
        super(Block, self).__init__()
        self.layer = self._make_layer(inplanes, planes, f, c)
        self.max_col = c
        self.max_dep = 2 ** (c-1)
        self.join_index = self._make_join_index(c)
        self.num_col = self._make_num_col(c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _make_layer(self, inplanes, planes, f, c):
        num_col = c
        per_col = [2 ** (-i-1) for i in range(-c, 0)]
        # 按照每行每列把卷积层(conv+bn+relu)加进去
        layer = nn.ModuleList([nn.ModuleList() for col in range(num_col)])
        for col, col_mod in zip(per_col, layer):
            for p in range(col):
                if p == 0:
                    col_mod.append(f(inplanes, planes))
                else:
                    col_mod.append(f(planes, planes))
        return layer

    # 指示每一次的join操作是集合的那几列
    def _make_join_index(self, c):
        index = [2, 3]
        for i in range(c-3):
            index = index + index
            index[-1] += 1
        return index

    def _make_num_col(self, c):
        return [2**i for i in range(c)]

    def join(self, x):
        out = torch.stack(x)
        out = out.mean(dim=0)
        return out

    def forward(self, x):
        # 每一列输入的数据
        in_x = [x for i in range(self.max_col)]
        # 索引每一列的卷积层编号
        num_col = [0 for i in range(self.max_col-1)]
        index = 0
        # 遍历每一层join操作
        for i in range(self.max_dep // 2):
            join_x_index = self.join_index[index]
            for j in range(join_x_index):
                if j == 0:
                    in_x[0] = self.layer[0][i*2+1](self.layer[0][i*2](in_x[0]))
                else:
                    in_x[j] = self.layer[j][num_col[j-1]](in_x[j])
            to_cat = in_x[:join_x_index]
            cat_x = self.join(to_cat)
            for l in range(join_x_index):
                in_x[l] = cat_x
                if l != 0:
                    num_col[l-1] += 1
        cat_x = self.pool(cat_x)
        return cat_x


# 论文里的image net版本
class FractalNet(nn.Module):
    def __init__(self, c, b, num_class=1000):
        super(FractalNet, self).__init__()
        self.inplanes = 64
        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.subnet = self._make_block(Block, Conv, c, b)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_class)

    def _make_block(self, block, f, c, b):
        sub_net = list()
        for i in range(b):
            sub_net.append(block(self.inplanes, self.inplanes * block.ex, f, c))
            self.inplanes = self.inplanes * block.ex
        return nn.Sequential(*sub_net)

    def forward(self, x):
        x = self.beginning(x)
        x = self.subnet(x)
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


def fractalnet34(num_class=1000, initialize=False):
    net = FractalNet(4, 4, num_class)
    if initialize:
        net.initialization()
    return net