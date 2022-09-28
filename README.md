# MobileViTv3 : Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features

This repository contains MobileViTv3's source code for training and evaluation and is inspired by MobileViT ([paper](https://arxiv.org/abs/2110.02178?context=cs.LG), [code](https://github.com/apple/ml-cvnets)).

## Installation and Training models:
We recommend to use Python 3.8+ and [PyTorch](https://pytorch.org) (version >= v1.8.0) with `conda` environment.
For setting-up the python environment with conda, see [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


### MobileViTv3\-S,XS,XXS
Download [MobileViTv1](https://github.com/apple/ml-cvnets/tree/d38a116fe134a8cd5db18670764fdaafd39a5d4f) and replace the files provided in [MobileViTv3-v1](MobileViTv3-v1).

Conda environment used for training: [environment_cvnet.yml](MobileViTv3-v1).

Then install according to instructions provided in the downloaded repository.
For training, use training-and-evaluation readme given in the downloaded repository.


### MobileViTv3\-1.0,0.75,0.5
Download [MobileViTv2](https://github.com/apple/ml-cvnets/tree/84d992f413e52c0468f86d23196efd9dad885e6f) and replace the files provided in [MobileViTv3-v2](MobileViTv3-v2).
Then install according to the instructions provided in the downloaded repository.

Conda environment used for training: [environment_mbvt2.yml](MobileViTv3-v2)

Then install according to instructions provided in the downloaded repository.
For training, use training-and-evaluation readme given in the downloaded repository.



## Trained models:

checkpoint\_ema\_best.pt files inside the model folder is used to generated the accuracy of models.

## Classification ImageNet-1K:
| Model name | Accuracy | foldername  |
| :---: | :---: | :---: |
| MobileViTv3\-S | 79.3 | mobilevitv3\_S\_e300\_7930 |
| MobileViTv3\-XS | 76.7 | mobilevitv3\_XS\_e300\_7671 |
| MobileViTv3\-XXS | 70.98 | mobilevitv3\_XXS\_e300\_7098 |
| MobileViTv3\-1.0 | 78.64 | mobilevitv3\_1\_0\_0 |
| MobileViTv3\-0.75 | 76.55 | mobilevitv3\_0\_7\_5 |
| MobileViTv3\-0.5 | 72.33 | mobilevitv3\_0\_5\_0 |

## Segmentation PASCAL VOC 2012:
| Model name | mIoU | foldername  |
| :---: | :---: | :---: |
| MobileViTv3\-S | 79.59 | mobilevitv3\_S\_voc\_e50\_7959 |
| MobileViTv3\-XS | 78.77 | mobilevitv3\_XS\_voc\_e50\_7877 |
| MobileViTv3\-XXS | 74.01 | mobilevitv3\_XXS\_voc\_e50\_7404 |
| MobileViTv3\-1.0 | 80.04 | mobilevitv3\_voc\_1\_0\_0 |
| MobileViTv3\-0.5 | 76.48 | mobilevitv3\_voc\_0\_5\_0 |

## Segmentation ADE20K:
| Model name | mIoU | foldername  |
| :---: | :---: | :---: |
| MobileViTv3\-1.0 | 39.13 | mobilevitv3\_ade20k\_1\_0\_0 |
| MobileViTv3\-0.75 | 36.43 |mobilevitv3\_ade20k\_0\_7\_5  |
| MobileViTv3\-0.5 | 39.13 | mobilevitv3\_ade20k\_0\_5\_0 |

## Detection COCO:
| Model name | mAP | foldername  |
| :---: | :---: | :---: |
| MobileViTv3\-S | 27.3 | mobilevitv3\_S\_coco\_e200\_2730 |
| MobileViTv3\-XS | 25.6 | mobilevitv3\_XS\_coco\_e200\_2560 |
| MobileViTv3\-XXS | 19.3 | mobilevitv3\_XXS\_coco\_e200\_1930 |
| MobileViTv3\-1.0 | 27.0 | mobilevitv3\_coco\_1\_0\_0 |
| MobileViTv3\-0.75 | 25.0 | mobilevitv3\_coco\_0\_7\_5 |
| MobileViTv3\-0.5 | 21.8 | mobilevitv3\_coco\_0\_5\_0 |


## Citation

MobileViTv3 paper reference will be added soon.
