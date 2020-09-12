## Learning Efﬁcient GANs using Differentiable Masks and Co-Attention Distillation 

<div align=center><img src="img/framework.png" height = "50%" width = "60%"/></div>

Framework of our method. We first build a pre-trained model similar to a GAN network, upon which a differentiable mask is imposed to scale the convolutional outputs of the generator and derive a light-weight one. Then, the co-Attention of the pre-trained GAN and the outputs of the last-layer convolutions of the discriminator are distilled to stabilize the training of the light-weight model.



#### CycleGAN

```shell
python train.py 
--dataroot ../datasets/horse2zebra/ 
--model cyclegan 
--gpu_ids 3 
--display_id 30 
--name horse2zebra_adaptive1_100relutripletbound0.0008 
--n_epochs 100 
--n_epochs_decay 100 

--mask 
--mask_weight_decay 0.0008 
--lambda_update_coeff 1.0 
--block_coeff 1.0 
--upconv_coeff 100.0 
--upconv_bound 
```

#### Pix2Pix

```shell
python train.py 
--dataroot ../datasets/edges2shoes/ 
--model pix2pix 
--name mask_pix2pix_edges2shoes  
--gpu_ids 0 
--display_id 1
--n_epochs 5 
--n_epochs_decay 15 
--load_size 256 
--no_flip 
--batch_size 4
--ngf 64 
--ndf 128 
--gan_mode vanilla/hinge 

--mask  
--mask_weight_decay 1e-3 
--lambda_update_coeff 1.0

python train.py  
--dataroot ../datasets/cityscapes/ 
--model pix2pix 
--name cityscape_pretrain_hinge 
--gpu_ids 1 
--display_id 10 
--n_epochs 100 
--n_epochs_decay 150 
--load_size 256 
--no_flip 
--ngf 64 
--ndf 128 
--gan_mode vanilla/hinge
--direction BtoA 

--mask  
--mask_weight_decay 1e-3 
--lambda_update_coeff 1.0
```

#### Test

```shell
python test.py 
--dataroot ../datasets/horse2zebra/ 
--name test
--model cyclegan 
--mask 
--gpu_ids 0
--load_path ./experiments/horse2zebracube_adaptive1_100relutripletbound0.0008/model_best.pth
```

#### Prune

```shell
python prune.py
--dataroot ../datasets/horse2zebra/ 
--model cyclegan 
--mask 
--checkpoints_dir ./experiments/horse2zebracube_adaptive1_100relutripletbound0.0008
--name attention
--load_path ./experiments/horse2zebracube_adaptive1_100relutripletbound0.0008/model_best.pth
--gpu_ids 0 
--display_id 1

--finetune
--lambda_attention_distill 100.0 
--lambda_discriminator_distill 0.001
--pretrain_path ../pretrain/horse2zebra_pretrain.pth
```

#### Continue Train

```shell
python train.py 
--dataroot ../datasets/horse2zebra/ 
--model cyclegan 
--gpu_ids 3 
--display_id 30 
--name horse2zebracube_adaptive1_100relutripletbound0.0008 
--n_epochs 100 
--n_epochs_decay 100 
--continue_train True 
--epoch_count 66 
--load_path ./experiments/horse2zebracube_adaptive1_100relutripletbound0.0008/checkpoints/model_65.pth

--mask 
--mask_weight_decay 0.0008 
--lambda_update_coeff 1.0 
--block_coeff 1.0 
--upconv_coeff 100.0 
--upconv_bound 
```

#### Other parameter

```shell
optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to images (should have subfolders trainA, trainB,
                        valA, valB, etc)
  --name NAME           name of the experiment. It decides where to store
                        samples and models
  --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU
  --checkpoints_dir CHECKPOINTS_DIR
                        models are saved here
  --phase PHASE         train, val, test, etc. default:train
  --load_path LOAD_PATH
                        The path of load model. default:None
  --use_pretrain_d USE_PRETRAIN_D
                        Using pretrain-D model to finetune new model.
                        default:False
  --pretrain_path PRETRAIN_PATH
                        The path of pretrain model. defalut:None
  --model MODEL         chooses which model to use. [cyclegan | pix2pix].
                        default:cyclegan
  --input_nc INPUT_NC   # of input image channels: 3 for RGB and 1 for
                        grayscale. default:3
  --output_nc OUTPUT_NC
                        # of output image channels: 3 for RGB and 1 for
                        grayscale. default:3
  --ngf NGF             # of gen filters in the last conv layer. default:64.
  --ndf NDF             # of discrim filters in the first conv layer.
                        default:64.
  --no_dropout          no dropout for the generator
  --mask                use MaskedGenerator. default:False
  --mask_weight_decay MASK_WEIGHT_DECAY
                        weight decay for mask_weight. default:0.0
  --mask_loss_type MASK_LOSS_TYPE
                        The type of mask loss decay [bound | exp | relu].
                        default:relu
  --unmask_last_upconv UNMASK_LAST_UPCONV
                        Unmask last upconv or not. default:False
  --update_bound_rule UPDATE_BOUND_RULE
                        The rule of update bound in mask layer. default:cube
  --continue_train CONTINUE_TRAIN
                        continue training: load the latest model
  --dataset_mode DATASET_MODE
                        chooses how datasets are loaded. [unaligned | aligned]
  --direction DIRECTION
                        AtoB or BtoA
  --serial_batches      if true, takes images in order to make batches,
                        otherwise takes them randomly
  --num_threads NUM_THREADS
                        # threads for loading data. default:8
  --batch_size BATCH_SIZE
                        input batch size. default:1
  --load_size LOAD_SIZE
                        scale images to this size. default:286
  --crop_size CROP_SIZE
                        then crop to this size. default:256
  --max_dataset_size MAX_DATASET_SIZE
                        Maximum number of samples allowed per dataset. If the
                        dataset directory contains more than max_dataset_size,
                        only a subset is loaded.
  --preprocess PREPROCESS
                        scaling and cropping of images at load time
                        [resize_and_crop | crop | scale_width |
                        scale_width_and_crop | none]
  --no_flip             if specified, do not flip the images for data
                        augmentation
  --display_winsize DISPLAY_WINSIZE
                        display window size for both visdom and HTML
  --block_coeff BLOCK_COEFF
  --upconv_coeff UPCONV_COEFF
  --upconv_solo         Only last upconv using update bound
  --display_freq DISPLAY_FREQ
                        frequency of showing training results on screen
  --display_ncols DISPLAY_NCOLS
                        if positive, display all images in a single visdom web
                        panel with certain number of images per row.
  --display_id DISPLAY_ID
                        window id of the web display
  --display_server DISPLAY_SERVER
                        visdom server of the web display
  --display_env DISPLAY_ENV
                        visdom display environment name (default is "main")
  --display_port DISPLAY_PORT
                        visdom port of the web display
  --update_html_freq UPDATE_HTML_FREQ
                        frequency of saving training results to html
  --print_freq PRINT_FREQ
                        frequency of showing training results on console
  --no_html             do not save intermediate training results to
                        [opt.checkpoints_dir]/[opt.name]/web/
  --save_epoch_freq SAVE_EPOCH_FREQ
                        frequency of saving checkpoints at the end of epochs.
                        default:1
  --epoch_count EPOCH_COUNT
                        the starting epoch count, we save the model by
                        <epoch_count>, <epoch_count>+<save_latest_freq>, ...
                        default:1
  --n_epochs N_EPOCHS   number of epochs with the initial learning rate.
                        default:100
  --n_epochs_decay N_EPOCHS_DECAY
                        number of epochs to linearly decay learning rate to
                        zero. default:100
  --lr LR               initial learning rate for adam. default:0.0002
  --gan_mode GAN_MODE   the type of GAN objective. [vanilla| lsgan | hinge |
                        wgangp]. vanilla GAN loss is the cross-entropy
                        objective used in the original GAN paper.
  --pool_size POOL_SIZE
                        the size of image buffer that stores previously
                        generated images. default:100
  --lr_policy LR_POLICY
                        learning rate policy. [linear | step | plateau |
                        cosine]
  --lr_decay_iters LR_DECAY_ITERS
                        multiply by a gamma every lr_decay_iters iterations
  --lambda_A LAMBDA_A   weight for cycle loss (A -> B -> A) default:10.0
  --lambda_B LAMBDA_B   weight for cycle loss (B -> A -> B) default:10.0
  --lambda_identity LAMBDA_IDENTITY
                        use identity mapping. Setting lambda_identity other
                        than 0 has an effect of scaling the weight of the
                        identity mapping loss. For example, if the weight of
                        the identity loss should be 10 times smaller than the
                        weight of the reconstruction loss, please set
                        lambda_identity = 0.1. default:0.5
  --lambda_L1 LAMBDA_L1
                        weight for L1 loss. default:100.0
  --lambda_group_lasso LAMBDA_GROUP_LASSO
                        weight for group lasso to sparsity resblock.
                        default:0.00
  --lambda_distill LAMBDA_DISTILL
                        weight for distill attention. default:0
  --lambda_d LAMBDA_D   weight for distill discriminator's feature map.
                        default:0.0
  --attention_normal ATTENTION_NORMAL
                        normalize for attention map
  --threshold THRESHOLD
                        The threshold of the removing block. default:0
  --upconv_bound        bound loss for upconv's mask weight
  --upconv_relu         contray relu loss for upconv's mask weight
  --lambda_update_coeff LAMBDA_UPDATE_COEFF
                        weight for update block's sparsity coeff after every
                        epoch training
  --ntest NTEST         # of test examples.
  --aspect_ratio ASPECT_RATIO
                        aspect ratio of result images
  --drn_path DRN_PATH   the path of drm model for mAP computation.
                        default:~/pretrain/drn-d-105_ms_cityscapes.pth
  --finetune            Finetune after prune
  --scratch             Finetune from scratch
```