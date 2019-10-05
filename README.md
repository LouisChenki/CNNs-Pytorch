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

##Usage
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
