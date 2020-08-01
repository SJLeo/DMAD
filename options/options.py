import argparse
import os

import utils.util as util

parser = argparse.ArgumentParser('DMAD')

# basic parameters
parser.add_argument('--dataroot', required=True,
                    help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--name', type=str, default='default', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./experiments', help='models are saved here')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc. default:train')
parser.add_argument('--load_path', type=str, default=None, help='The path of load model. default:None')
parser.add_argument('--use_pretrain_d', type=bool, default=False, help='Using pretrain-D model to finetune new model. default:False')
parser.add_argument('--pretrain_path', type=str, default=None, help='The path of pretrain model. defalut:None')

# model parameters
parser.add_argument('--model', type=str, default='cyclegan',
                    help='chooses which model to use. [cyclegan | pix2pix]. default:cyclegan')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale. default:3')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale. default:3')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer. default:64.')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer. default:64.')
# parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
# parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
# parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
# parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
# parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
# parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
parser.add_argument('--mask', action='store_true', help='use MaskedGenerator. default:False')
# parser.add_argument('--mask_init_gain', type=float, default=0.1, help='scaling factor for init mask_weight. default:0.1')
parser.add_argument('--mask_weight_decay', type=float, default=0.0, help='weight decay for mask_weight. default:0.0')
parser.add_argument('--mask_loss_type', type=str, default='relu', help='The type of mask loss decay [bound | exp | relu]. default:relu')
# parser.add_argument('--mask_grad_clip', type=bool, default=False, help='clip mask grad. default:False')
parser.add_argument('--unmask_last_upconv', type=bool, default=False, help='Unmask last upconv or not. default:False')
parser.add_argument('--update_bound_rule', type=str, default='cube', help='The rule of update bound in mask layer. default:cube')
# parser.add_argument('--threshold', type=float, default=0.01, help='The threshold for pruning. default:0.01')
parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')
# parser.add_argument('--combine_bestmodel', type=bool, default=True, help='Combine the best AtoB model and best BtoA model')

# dataset parameters
parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned]')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data. default:8')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size. default:1')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size. default:286')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size. default:256')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')

# additional parameters
parser.add_argument('--block_coeff', type=float, default=1.0)
parser.add_argument('--upconv_coeff', type=float, default=1.0)
parser.add_argument('--upconv_solo', action='store_true', help='Only last upconv using update bound')
# parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
# parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
# parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
# parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

# train parameter
parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

# parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
# parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs. default:1')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ... default:1')

parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate. default:100')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero. default:100')
# parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam. default:0.0002')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | hinge | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=100, help='the size of image buffer that stores previously generated images. default:100')
parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A) default:10.0')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B) default:10.0')
parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1. default:0.5')
parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss. default:100.0')
# parser.add_argument('--group_lasso', type=bool, default=False, help='group lasso for sparsity resblock. default:False')
parser.add_argument('--lambda_group_lasso', type=float, default=0.0, help='weight for group lasso to sparsity resblock. default:0.00')
parser.add_argument('--lambda_attention_distill', type=float, default=0.0, help='weight for distill attention. default:0')
parser.add_argument('--lambda_discriminator_distill', type=float, default=0.0, help='weight for distill discriminator\'s feature map. default:0.0')
# parser.add_argument('--solo', type=bool, default=False, help='add Discriminator attention map. defualt:False')
parser.add_argument('--attention_normal', type=bool, default=True, help='normalize for attention map')
parser.add_argument('--threshold', type=int, default=0, help='The threshold of the removing block. default:0')
parser.add_argument('--upconv_bound', action='store_true', help='bound loss for upconv\'s mask weight')
parser.add_argument('--upconv_relu', action='store_true', help='contray relu loss for upconv\'s mask weight')
parser.add_argument('--bound_loss_index', type=str, default='000000000000000', help='use bound loss for conv/upconv, 1 stand for using. default:000000000000000')
parser.add_argument('--lambda_update_coeff', type=float, default=0.0, help='weight for update block\'s sparsity coeff after every epoch training')

#test parameter
parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
# parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
parser.add_argument('--drn_path', type=str, default='/home/lishaojie/pretrain/drn-d-105_ms_cityscapes.pth', help='the path of drm model for mAP computation. default:~/pretrain/drn-d-105_ms_cityscapes.pth')

# parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
# parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

# prune parameter
parser.add_argument('--finetune', action='store_true', help='Finetune after prune')
parser.add_argument('--scratch', action='store_true', help='Finetune from scratch')

# early stop
parser.add_argument('--AtoB_macs_threshold', type=float, default=0, help='early stop macs threshold used in netG_A')
parser.add_argument('--BtoA_macs_threshold', type=float, default=0, help='early stop macs threshold used in netG_B')

def print_options(opt, parser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    if opt.name == 'default':
        opt.name = opt.update_bound_rule + '_'
        if opt.lambda_group_lasso > 0:
            opt.name += 'group' + str(opt.lambda_group_lasso)
        opt.name += 'solo' + str(opt.block_coeff) + '_' + str(opt.upconv_coeff)
        opt.name += opt.mask_loss_type + str(opt.mask_weight_decay) + '_'

        opt.name += 'distill' + str(opt.lambda_attention_distill)

        if opt.unmask_last_upconv:
            opt.name += '_unmask'

        if opt.use_pretrain_d:
            opt.name += '_pretrainD'

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'config.txt'.format(opt.phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

def parse():

    opt = parser.parse_args()
    print_options(opt, parser)
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    bound_loss_index_list = list(opt.bound_loss_index)
    opt.bound_loss_index = []
    for index in bound_loss_index_list:
        if index == '1':
            opt.bound_loss_index.append(True)
        else:
            opt.bound_loss_index.append(False)
    return opt