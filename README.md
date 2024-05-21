# Distribution Alignment Optimization through Neural Collapse for Long-tailed Classification (ICML, 2024)

by Jintong Gao<sup>1</sup>, He Zhao<sup>2</sup>, Dandan Guo<sup>1</sup>, Hongyuan Zha<sup>3</sup>, 

<sup>1</sup>Jilin University, <sup>2</sup>CSIRO's Data61, <sup>3</sup>The Chinese University of Hong Kong, Shenzhen

This is the official implementation of [Distribution Alignment Optimization through Neural Collapse for Long-tailed Classification](https://openreview.net/pdf?id=Hjwx3H6Vci) in PyTorch.

## Requirements:

All codes are written by Python 3.8 with 

```
PyTorch >=1.5
torchvision >=0.6
TensorboardX 1.9
Numpy 1.17.3
```

## Training

To train the model(s) in the paper, run this command:

### CIFAR-LT

CIFAR-10-LT (CE-DRW + DisA):

```
python cifar_train.py --dataset cifar10 --num_classes 10 --loss_type CE --train_rule DRW --lamda 0.1 --gpu 0
```

CIFAR-100-LT (CE-DRW + DisA):

```
python cifar_train.py --dataset cifar100 --num_classes 100 --loss_type CE --train_rule DRW --lamda 0.1 --gpu 0
```

## Evaluation

To evaluate my model, run:

```
python test.py --dataset cifar10 --num_classes 10 --gpu 0 --resume model_path
```
## Citation

If you find our paper and repo useful, please cite our paper.

```
@inproceedings{
Gao2024DisA,
title={Distribution Alignment Optimization through Neural Collapse for Long-tailed Classification},
author={Jintong Gao and He Zhao and Dandan Guo and Hongyuan Zha},
booktitle={International Conference on Machine Learning (ICML)},
year={2024}
}
```
## Acknowledgement

[ETF-DR](https://github.com/NeuralCollapseApplications/ImbalancedLearning)

[INC](https://github.com/Pepper-lll/NCfeature)

[RBL](https://gitee.com/gaopeifeng/rbl)

## Contact

If you have any questions when running the code, please feel free to concat us by emailing

+ Jintong Gao ([gaojt20@mails.jlu.edu.cn](mailto:gaojt20.mails.jlu.edu.cn))
