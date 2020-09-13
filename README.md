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
  |  horse2zebra  | [Link](https://drive.google.com/drive/folders/1wUGazdIe_B4gHs_gMq-jRWW53yKbyOcs?usp=sharing) |
  | summer2winter | [Link](https://drive.google.com/drive/folders/1JKJlpUDdD4TdXdwPwfdWUiF4PsXLAbto?usp=sharing) |

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

### Compressed Models

We provide our compressed models in the experiments.

|   Model   |     Task      | MACs(Compress Rate) | Parameters(Compress Rate) |  FID/mIOU  |                           Download                           |
| :-------: | :-----------: | :-----------------: | :-----------------------: | :--------: | :----------------------------------------------------------: |
| CycleGAN  |  horse2zebra  |    3.97G(14.3×)     |       0.42M(26.9×)        | FID:62.41  | [Link](https://drive.google.com/file/d/1juofKZh6si3_oBNUKGRLatRFgcz4g6GL/view?usp=sharing) |
| CycleGAN* |  horse2zebra  |    2.41G(23.6×)     |       0.28M(40.4×)        | FID:62.96  | [Link](https://drive.google.com/file/d/1yyU5QeAj9I3lDe_e5ZXiEa2uJIyIFyBN/view?usp=sharing) |
| CyclceGAN |  zebra2horse  |    3.50G (16.2×)    |       0.30M (37.7×)       | FID:139.3  | [Link](https://drive.google.com/file/d/1juofKZh6si3_oBNUKGRLatRFgcz4g6GL/view?usp=sharing) |
| CyclceGAN | summer2winter |    3.18G (17.9×)    |       0.24M (47.1×)       | FID:78.24  | [Link](https://drive.google.com/file/d/1EOtnr1viTWZxnemh7PxNQYp9k6qBBydk/view?usp=sharing) |
| CyclceGAN | winter2summer |    4.29G (13.2×)    |       0.45M (25.1×)       | FID:70.97  | [Link](https://drive.google.com/file/d/1EOtnr1viTWZxnemh7PxNQYp9k6qBBydk/view?usp=sharing) |
|  Pix2Pix  |  edges2shoes  |    2.99G (6.22×)    |       2.13M (25.5×)       | FID:46.95  | [Link](https://drive.google.com/file/d/1o9DqyxrXTHVviAKAq0dkRwZoicbYZsPg/view?usp=sharing) |
| Pix2Pix*  |  edges2shoes  |    4.30G (4.32×)    |      0.54M (100.7×)       | FID:24.08  | [Link](https://drive.google.com/file/d/1eDQJZXS2Vctt_8uPr1tZfOZOvsZFO-1x/view?usp=sharing) |
|  Pix2Pix  |  cityscapes   |    3.96G (4.70×)    |       1.73M (31.4×)       | mIOU:40.53 | [Link](https://drive.google.com/file/d/1Yh7Hn28cEk8A19HaH_hALpskmulREvPF/view?usp=sharing) |
| Pix2Pix*  |  cityscapes   |    4.39G (4.24×)    |       0.55M (98.9×)       | mIOU:41.47 | [Link](https://drive.google.com/file/d/1g-GtEc_ev8yjpyjeqqDMgqnRKOYF3r3M/view?usp=sharing) |

[^*]: indicates that a generator with separable convolu-tions is adopted

You can use the following code to test our compression models.

```shell
python test.py 
	--dataroot ./database/horse2zebra
	--model cyclegan
	--load_path ./result/horse2zebra.pth
```

### Tips

Any problem, free to contact the authors via emails:[shaojieli@stu.xmu.edu.cn](mailto:shaojieli@stu.xmu.edu.cn) or [lmbxmu@stu.xmu.edu.cn](mailto:lmbxmu@stu.xmu.edu.cn) .