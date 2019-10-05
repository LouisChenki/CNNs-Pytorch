# CNNs-Pytorch
Convolutional Neural Network of classification models on Pytorch(to be continue)  

![](https://img.shields.io/badge/Python-Pytorch-orange.svg?style=flat-square) 
![](https://img.shields.io/badge/CNNs-Vision-blue.svg?style=flat-square) 
![](https://img.shields.io/badge/By-KaiQiChen-red.svg?style=flat-square)


## Introduction
In the past few years, in the field of computer vision,**Convolutional Neural Networks(CNNs)** have developed rapidly, especially in the image classification task. This project is based on the recent computer vision top conferences (CVPR, ICCV, ECCV) and other excellent papers. Various types of convolutional neural networks are implemented on the framework of **Pytorch**. Some models have special training, regularization, test mode, etc., because the authors of the paper are more brief, and because the author of this project is relatively simple, there may be minor problems. It is proposed that the project will continue to be updated.

## Requirements
This is my experiment eviroument  

**1.hardware:**  

- Intel@Core i9-9900K CPU @ 3.60HZ x 16
- GeForce RTX 2080 Ti x1
- 32 GB DDR4

**2.software:**

- Python 3.7.3
- Pytorch 1.1.0
- CUDA 10

## Usage
**1.dataset**

By default, the code uses **cifar10** dataset from torchvision for model training, and can be replaced with its own dataset.

**2.train**

a.You need to specify the <font color=CornflowerBlue>net</font> you want to train using arg `-net`  
b.You need to specify the  <font color=CornflowerBlue>number of labels</font> you want to train using arg `-num_class` 
c.You need to specify Whether to <font color=CornflowerBlue>initialize the weight</font> you want to train using arg `-initialize`
d.You need to specify the <font color=CornflowerBlue>learning rate</font> you want to train using arg `-lr`

For example
```bash
$ python test.py -net resnet18 -num_class 10 -initialize True -lr 0.001
```

**3.models and papers**

The convolutional neural network **model** and **paper** contained in this project are as follows:

---
* AlexNet
* Paper: [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
---

* VGG 11
* vgg 11 lrn
* VGG 13
* VGG 16C
* VGG 16D
* VGG19
* Paper:[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
---
* GoogLeNet
* Paper:[Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)
---
* ResNet 18
* ResNet 34
* ResNet 50
* ResNet 101
* ResNet 152
* Paper:[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
---
* Fractalnet 34
* Paper:[FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/pdf/1605.07648v1.pdf)
---
* Inception v3
* Paper:[Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
* Paper:[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567v3.pdf)
---
* MobileNet v1
* Paper:[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
---
* PreActResNet 18
* PreActResNet 34
* PreActResNet 50
* PreActResNet 101
* PreActResNet 152
* Paper:[Identity mappings in deep residual networks](https://arxiv.org/pdf/1603.05027.pdf)
---
* SENet 50
* SENet 101
* SENet 152
* Paper:[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
---
* ShuffleNet 0.5x g1
* ShuffleNet 0.5x g2
* ShuffleNet 0.5x g3
* ShuffleNet 0.5x g4
* ShuffleNet 1x g1
* ShuffleNet 1x g2
* ShuffleNet 1x g3
* ShuffleNet 1x g4
* ShuffleNet 0.25x g1
* ShuffleNet 0.25x g2
* ShuffleNet 0.25x g3
* ShuffleNet 0.25x g4
* Paper:[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)
---
* SqueezeNet
* Paper:[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf)
---
* StochasticDepth 18
* StochasticDepth 34
* StochasticDepth 50
* StochasticDepth 101
* StochasticDepth 152
* Paper:[Deep networks with stochastic depth](https://arxiv.org/pdf/1603.09382.pdf)
---
* Wide ResNet 40 4
* Wide ResNet 16 8
* Wide ResNet 40 4
* Wide ResNet 28 10
* Paper:[Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf)
---
* Xception
* Paper:[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf)
---
* **to be continue**
