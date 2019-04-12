# Shufflenet-v2-Pytorch

## Introduction

This is a Pytorch implementation of faceplusplus's ShuffleNet-v2. For details, please read the following papers:

[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## Pretrained Models on ImageNet

We provide pretrained ShuffleNet-v2 models on ImageNet,which achieve slightly better accuracy rates than the original ones reported in the paper.

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):

| Network            | Top-1  | Top-1 (reported in the paper) |
| ------------------ | ------ | ----------------------------- |
| ShuffleNet-v2-x0.5 | 60.646 | 60.3                          |
| ShuffleNet-v2-x1   | 69.402 | 69.4                          |


## Evaluate Models

```
python3 eval.py --arch=shufflenetv2_x0_5 --pretrained ./ILSVRC2012/
```

```
python3 eval.py --arch=shufflenetv2_x1_0 --pretrained ./ILSVRC2012/
```

## Version:

- Python3.6.X
- torch 1.0.1

Dataset prepare Refer to https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset

