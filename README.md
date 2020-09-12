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

- Train the model using our differentiable masks (eg. horse2zebra)

  ```shell
  bash scripts/cyclegan/horse2zebra/train.sh
  ```

- Finetune the searched light-weight models with co-Attention distillation

  ```shell
  bash scripts/cyclegan/horse2zebra/finetune.sh
  ```

#### Pix2Pix

- Train the model using our differentiable masks (eg. cityscapes)

  ```shell
  bash scripts/pix2pix/cityscapes/train.sh
  ```

- Finetune the searched light-weight models with co-Attention distillation

  ```shell
  bash scripts/pix2pix/cityscapes/finetune.sh
  ```