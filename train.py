import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import network


# 数据下载与转换
batchsize = 64
max_test_acc = 1
pretrain = True
acc_judge = 0.70
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-num_class', type=int, required=True, help='num of class')
    parser.add_argument('-initialize', type=bool, required=True, help='initialize')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = network(args)

    device = torch.device('cuda:0')
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))

    # 保存中间参数
    acc_test_epoch, loss_epoch = [], []
    acc_train_epoch = []
    x = []

    for epoch in range(70):
        runing_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            runing_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d]  accuracy:%.3f %%  loss: %.3f' % (epoch + 1, i + 1, 100 * (correct / total), runing_loss / 200))
                runing_loss = 0
        loss_epoch.append(runing_loss)

        # 跑完一个epoch测试一下test
        print('waiting test')
        with torch.no_grad():
            correct = 0
            total = 0
            # 将模型设置为evaluation，防止Dropout对测试结果的影响
            net.eval()
            # 因为testset也是分了batch的，所以进行循环
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += torch.tensor(labels.size(0))
                correct += (predicted == labels).sum()
            acc_test_tensor = (100 * correct.float() / total)
            acc_test = float(acc_test_tensor.item()) / 100
            print('The Accuracy of the Test is: %.3f%%' % (100 * acc_test))
            acc_test_epoch.append(acc_test)

            # 训练数据集精度需要在网络的bn层关闭的情况下再计算一次
            correct = 0
            total = 0
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += torch.tensor(labels.size(0))
                correct += (predicted == labels).sum()
            acc_train_tensor = (100 * correct.float() / total)
            acc_train = float(acc_train_tensor.item()) / 100
            print('The Accuracy of the Train is: %.3f%%' % (100 * acc_train))
            # 网络转回为训练模式
            net.train()
            acc_train_epoch.append(acc_train)

            # x 储存acc_test_epo 与 acc_test_epoch 相应的iter数
            x.append((epoch+1) * len(trainloader))

            # 大于一定程度的测试精确度就可以保存模型
            if acc_test >= acc_judge:
                stat = {
                    'epoch': epoch,
                    'optimizer': optimizer,
                    'loss': runing_loss,
                    'acc_test': acc_test,
                    'batchsize': batchsize,
                    'acc_train_epoch': acc_train_epoch,
                    'loss_epoch': loss_epoch,
                    'acc_test_epoch': acc_test_epoch,
                    'x': x
                }
                '''
                torch.save(stat, 'diresnet.pth')
                '''
                print('save')
                acc_judge = acc_test
            if acc_test > max_test_acc:
                print('the max of test accurancy is %f' % (max_test_acc))
                break

    print('Finished Training')