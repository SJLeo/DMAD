#### CycleGAN

```shell
python train.py 
--dataroot ../datasets/horse2zebra/ 
--model cyclegan 
--mask True 
--mask_weight_decay 0.002 
--lambda_update_coeff 2.0 
--block_coeff 1.0 
--upconv_coeff 100.0 
--upconv_bound 
--gpu_ids 3 
--display_id 30 
--name cube_adaptive2_100relutripletbound0.002 
--n_epochs 100 
--n_epochs_decay 100
```

#### Pix2Pix

```shell
python train.py 
--dataroot /media/disk2/lishaojie/GAN-Compresss/datasets/edges2shoes/ 
--name mask_pix2pix 
--model pix2pix 
--mask True 
--mask_weight_decay 1e-3 
--mask_init_gain 0.1
--update_bound_rule edges2shoes 
--mask_loss_type bound 
--gpu_ids 0 
--n_epochs 5 
--n_epochs_decay 15 
--load_size 256 
--no_flip 
--batch_size 4
--ngf 64
--save_epoch_freq 1 
--display_id 1
--lr 0.01
--lr_decay linear

python train.py  
--dataroot ../datasets/cityscapes/ 
--model pix2pix 
--n_epochs 100 
--n_epochs_decay 150 
--ngf 96 
--ndf 128 
--gpu_ids 1 
--display_id 10 
--gan_mode hinge 
--name cityscape_pretrain_hinge 
--direction BtoA
```

#### Test

```shell
python test.py 
--dataroot /media/disk2/lishaojie/GAN-Compresss/datasets/horse2zebra/ 
--name mask_cyclegan 
--model cyclegan 
--mask True 
--load_path ./experiments/new_cyclegan/checkpoints/model_best.pth
```

#### Prune

```shell
python prune.py
--dataroot /media/disk2/lishaojie/GAN-Compresss/datasets/horse2zebra/ 
--checkpoints_dir ./experiments/horse2zebra3_relu1e-4_distill0.0_unmask
--name pruned
--model cyclegan 
--unmask_last_upconv True
--mask True 
--load_path ./experiments/new_cyclegan/checkpoints/model_best.pth

-finetune
-scratch
--n_epochs 50 
--n_epochs_decay 50
--lr 0.0001
```

#### Continue Train

```shell
python train.py 
--continue_train True
--epoch_count 6
--load_path ./experiments/new_cyclegan/checkpoints/model_5.pth
--dataroot /media/disk2/lishaojie/GAN-Compresss/datasets/horse2zebra/ 
--name mask_cyclegan 
--model cyclegan 
--mask True 
--mask_weight_decay 1e-3 
--mask_grad_clip False
--update_bound_rule horse2zebra 
--mask_loss_type bound 
--gpu_ids 0 
--n_epochs 100 
--n_epochs_decay 100
--save_epoch_freq 5 
--display_id 1
--lr 0.01
--lr_decay linear
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
  --mask MASK           use MaskedGenerator. default:False
  --mask_init_gain MASK_INIT_GAIN
                        scaling factor for init mask_weight. default:0.02
  --mask_weight_decay MASK_WEIGHT_DECAY
                        weight decay for mask_weight. default:1e-3
  --mask_loss_type MASK_LOSS_TYPE
                        The type of mask loss decay [bound | exp | relu].
                        default:bound
  --update_bound_rule UPDATE_BOUND_RULE
                        The rule of update bound in mask layer.
                        default:horse2zebra
  --threshold THRESHOLD
                        The threshold for pruning. default:0.01
  --continue_train CONTINUE_TRAIN
                        continue training: load the latest model
  --dataset_mode DATASET_MODE
                        chooses how datasets are loaded. [unaligned | aligned]
  --direction DIRECTION
                        AtoB or BtoA
  --serial_batches      if true, takes images in order to make batches,
                        otherwise takes them randomly
  --num_threads NUM_THREADS
                        # threads for loading data. default:4
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
                        default:5
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
  --gan_mode GAN_MODE   the type of GAN objective. [vanilla| lsgan | wgangp].
                        vanilla GAN loss is the cross-entropy objective used
                        in the original GAN paper.
  --pool_size POOL_SIZE
                        the size of image buffer that stores previously
                        generated images. default:50
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
  --ntest NTEST         # of test examples.
  --aspect_ratio ASPECT_RATIO
                        aspect ratio of result images
```

