## Learning Efﬁcient GANs using Differentiable Masks and Co-Attention Distillation 

<div align=center><img src="img/framework.png" height = "50%" width = "60%"/></div>

Framework of our method. We first build a pre-trained model similar to a GAN network, upon which a differentiable mask is imposed to scale the convolutional outputs of the generator and derive a light-weight one. Then, the co-Attention of the pre-trained GAN and the outputs of the last-layer convolutions of the discriminator are distilled to stabilize the training of the light-weight model.

### Getting Started

The code has been tested using Pytorch1.5.1 and CUDA10.2 on Ubuntu 18.04.

Please type the command 

```shell
pip install -r requirements.txt
```

to install dependencies.

#### CycleGAN

- Download the Cyclcegan dataset (eg. horse2zebra)

  ```shell
  bash datasets/download_cyclegan_dataset.sh horse2zebra
  ```

- Download our pre-prepared real statistic information for computing FID, and then copy them to the root directionary of dataset.

  |     Task      |                           Download                           |
  | :-----------: | :----------------------------------------------------------: |
  |  horse2zebra  | [Link](https://drive.google.com/file/d/1ZTcEPLm3y8I_1RBDTzPySEVVJltshFu0/view?usp=sharing, https://drive.google.com/file/d/1s6uYYsKHg2d3yRmsiFVX8JYiacLCzqeL/view?usp=sharing) |
  | summer2winter | [Link](https://drive.google.com/file/d/1G0Kt9SAFEPBtogdBWW8sXiC9on9dTrqK/view?usp=sharing, https://drive.google.com/file/d/1NpJ1sUkiilYiGCHydxr8UYg_DO4gaFVf/view?usp=sharing) |

- Train the model using our differentiable masks (eg. horse2zebra)

  ```shell
  bash scripts/cyclegan/horse2zebra/train.sh
  ```

- Finetune the searched light-weight models with co-Attention distillation

  ```shell
  bash scripts/cyclegan/horse2zebra/finetune.sh
  ```

#### Pix2Pix

- Download the Pix2Pix dataset (eg. edges2shoes)

  ```shell
  bash datasets/download_pix2pix_dataset.sh edges2shoes-r
  ```

- Download our  pre-trained real statistic information for computing FID or  DRN-D-105 model for computing mIOU, and then copy them to the root directionary of dataset.

  |    Task     |                           Download                           |
  | :---------: | :----------------------------------------------------------: |
  | edges2shoes | [Link](https://drive.google.com/file/d/1B2iBvJWuhlYYgR5wpjMnWoDJcD4NNK-p/view?usp=sharing) |
  | cityscapes  | [Link](https://drive.google.com/file/d/1V4RmILQ0QGNQTRvMlSzN-rkvRAdmKMOr/view?usp=sharing) |

- Train the model using our differentiable masks (eg. edges2shoes)

  ```shell
  bash scripts/pix2pix/edges2shoes/train.sh
  ```

- Finetune the searched light-weight models with co-Attention distillation

  ```shell
  bash scripts/pix2pix/edges2shoes/finetune.sh
  ```