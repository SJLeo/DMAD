import torch
import torch.nn as nn

from models.MaskLayer import Mask
from models.MobileCycleGAN import MobileCycleGANModel
import utils.util as util
from utils.image_pool import ImagePool
from models.GANLoss import GANLoss

import itertools
import functools
import os
import copy
import math
from collections import OrderedDict
from thop import profile

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=nn.InstanceNorm2d,
                 use_bias=True, scale_factor=1):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=use_bias),
            norm_layer(in_channels * scale_factor),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x):
        return self.conv(x)

class MaskMobileResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias, opt=None):
        super(MaskMobileResnetBlock, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=dim, out_channels=dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(dim),
            Mask(dim, mask_loss_type=self.opt.mask_loss_type),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SeparableConv2d(in_channels=dim, out_channels=dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(dim),
            Mask(dim, mask_loss_type=self.opt.mask_loss_type),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MaskMobileResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect', opt=None):
        assert (n_blocks >= 0)
        super(MaskMobileResnetGenerator, self).__init__()
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if self.opt.upconv_relu and not opt.upconv_solo:
            upconv_mask_loss_type = 'uprelu'
        elif self.opt.upconv_bound and not opt.upconv_solo:
            upconv_mask_loss_type = 'bound'
        else:
            upconv_mask_loss_type = 'relu'

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 Mask(ngf, mask_loss_type=upconv_mask_loss_type),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      Mask(ngf * mult * 2, mask_loss_type=self.opt.mask_loss_type if i != 0 else 'bound'),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):
            model += [MaskMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                        dropout_rate=dropout_rate,
                                        use_bias=use_bias, opt=self.opt)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      Mask(int(ngf * mult / 2), mask_loss_type=upconv_mask_loss_type if i != n_downsampling - 1 else 'bound'),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.block_sparsity_coeff = torch.FloatTensor([1.0] * (self.opt.ngf * 4)).to(self.device)

    def update_masklayer(self, bound):
        for module in self.modules():
            if isinstance(module, Mask):
                module.update(bound)

    def update_sparsity_factor(self):

        group_mask_weight_names = []
        group_mask_weight_names.append('model.11')
        for i in range(13, 22, 1):
            group_mask_weight_names.append('model.%d.conv_block.9' % i)

        group_weights = []
        bound = 0.0
        for name, module in self.named_modules():
            if name in group_mask_weight_names:
                bound = module.bound
                group_weights.append(module.mask_weight)

        group_weights = torch.stack(tuple(group_weights), dim=0)
        for i in range(group_weights.size(1)):

            lt_bound = sum(group_weights[:, i] < -bound)
            gt_bound = sum(group_weights[:, i] > bound)
            in_bound = group_weights.size(0) - (lt_bound + gt_bound)

            if gt_bound > 0 or lt_bound == group_weights.size(0):
                self.block_sparsity_coeff[i] = 0.0
            else:
                self.block_sparsity_coeff[i] = group_weights.size(0) / float(in_bound) * self.opt.lambda_update_coeff
        # print(self.block_sparsity_coeff)

    def print_sparse_info(self, logger):

        group_mask_weight_names = []
        group_mask_weight_names.append('model.11')
        for i in range(13, 22, 1):
            group_mask_weight_names.append('model.%d.conv_block.9' % i)

        residual_mask = [True for _ in range(self.opt.ngf * 4)]

        for name, module in self.named_modules():
            if isinstance(module, Mask):

                if name in group_mask_weight_names:
                    current_mask = module.mask_weight < -module.bound
                    for i in range(len(residual_mask)):
                        residual_mask[i] &= bool(current_mask[i])

                mask = module.get_current_mask()

                logger.info('%s sparsity ratio: %.2f\tone ratio: %.2f\t'
                      'total ratio: %.2f' % (name, float(sum(mask == 0.0)) / mask.numel(),
                                              float(sum(mask == 1.0)) / mask.numel(),
                                             (float(sum(mask == 0.0)) + float(sum(mask == 1.0))) / mask.numel()))
        logger.info('%s sparsity ratio: %.2f' % ('ResBlock', float(sum(residual_mask)) / len(residual_mask)))

    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    '''Defines a PatchGAN discriminator'''

    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):

        return self.model(x)

class MaskMobileCycleGANModel(nn.Module):

    def __init__(self, opt):
        super(MaskMobileCycleGANModel, self).__init__()
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'mask_weight']
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_B']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_A']
        self.visual_names = visual_names_A + visual_names_B

        self.netG_A = MaskMobileResnetGenerator(opt=self.opt, ngf=self.opt.ngf)
        self.netG_B = MaskMobileResnetGenerator(opt=self.opt, ngf=self.opt.ngf)

        self.netD_A = NLayerDiscriminator(ndf=self.opt.ndf)
        self.netD_B = NLayerDiscriminator(ndf=self.opt.ndf)
        self.init_net()

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        if self.opt.use_pretrain_d: # using pretrain model's discriminator to initial
            print('using pretrian discriminator')
            self.init_discriminator()

        self.group_mask_weight_names = []
        self.group_mask_weight_names.append('model.11')
        for i in range(13, 22, 1):
            self.group_mask_weight_names.append('model.%d.conv_block.9' % i)

        self.stop_AtoB_mask = False
        self.stop_BtoA_mask = False

        # define loss functions
        self.criterionGAN= GANLoss(opt.gan_mode).to(self.device)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()

        # define optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                            lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                            lr=opt.lr, betas=(0.5, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.schedulers = [util.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def set_input(self, input):

        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = [input['A_paths' if AtoB else 'B_paths'], input['B_paths' if AtoB else 'A_paths']]

    def forward(self):

        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(self.real_B)
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        mask_decay = self.opt.mask_weight_decay
        # Identity loss
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Mask weight decay loss
        self.loss_mask_weight = (self.get_mask_weight_loss(self.netG_A)
                                 + self.get_mask_weight_loss(self.netG_B)) * mask_decay
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + \
                      self.loss_mask_weight
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        # self.update_masklayer(epoch) # update mask alpha

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_models(self, epoch, save_dir, fid=None, isbest=False, direction='AtoB'):
        util.mkdirs(save_dir)
        ckpt = {
            'G_A': self.netG_A.state_dict(),
            'G_B': self.netG_B.state_dict(),
            'D_A': self.netD_A.state_dict(),
            'D_B': self.netD_B.state_dict(),
            'epoch': epoch,
            'fid': fid
        }
        if isbest:
            torch.save(ckpt, os.path.join(save_dir, 'model_best_%s.pth' % direction))
        else:
            torch.save(ckpt, os.path.join(save_dir, 'model_%d.pth' % epoch))

    def load_models(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device)
        self.netG_A.load_state_dict(ckpt['G_A'])
        self.netG_B.load_state_dict(ckpt['G_B'])
        self.netD_A.load_state_dict(ckpt['D_A'])
        self.netD_B.load_state_dict(ckpt['D_B'])

        print('loading the model from %s' % (load_path))
        return ckpt['fid'][0], ckpt['fid'][1]

    def init_net(self):
        self.netG_A.to(self.device)
        self.netG_B.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)

        util.init_weights(self.netG_A, init_type='normal', init_gain=0.02)
        util.init_weights(self.netG_B, init_type='normal', init_gain=0.02)
        util.init_weights(self.netD_A, init_type='normal', init_gain=0.02)
        util.init_weights(self.netD_B, init_type='normal', init_gain=0.02)

    def model_train(self):
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

    def model_eval(self):
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def get_current_visuals(self):
        """Return visualization images. """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def update_masklayer(self, current_iter, all_total_iters):

        self.netG_A.update_sparsity_factor()
        self.netG_B.update_sparsity_factor()

        update_bound_iters_count = all_total_iters * 0.75

        if current_iter > update_bound_iters_count:
            bound = 0.0
        else:
            if self.opt.update_bound_rule == 'cube':
                bound = 1 - math.pow(float(current_iter) / update_bound_iters_count, 1 / 3)
            else:
                bound = 1 - math.pow(float(current_iter) / update_bound_iters_count, 1 / 2)

            if bound < 0:
                bound = 0.0
        print('Bound: %.3f' % bound)

        self.early_stop_mask()

        if not self.stop_AtoB_mask:
            self.stable_weight(self.netG_A, bound=bound)
        else:
            print('AtoB early stop')
        if not self.stop_BtoA_mask:
            self.stable_weight(self.netG_B, bound=bound)
        else:
            print('BtoA early stop')

        self.netG_A.update_masklayer(bound if not self.stop_AtoB_mask else 0.0)
        self.netG_B.update_masklayer(bound if not self.stop_BtoA_mask else 0.0)

    def print_sparsity_info(self, logger):
        logger.info('netG_A')
        self.netG_A.print_sparse_info(logger)

        logger.info('netG_B')
        self.netG_B.print_sparse_info(logger)

    def get_mask_weight_loss(self, G):
        mask_weight_loss = 0.0
        for name, module in G.named_modules():

            if isinstance(module, Mask):
                if self.opt.lambda_update_coeff > 0 and name in self.group_mask_weight_names:
                    mask_weight_loss += module.get_block_decay_loss(G.block_sparsity_coeff)
                elif name == 'model.28' and self.opt.upconv_bound:
                    mask_weight_loss += module.get_weight_decay_loss() * self.opt.upconv_coeff
                elif  (name == 'model.24' or name == 'model.3' or name == 'model.7') and not self.opt.upconv_solo and self.opt.upconv_bound:
                    mask_weight_loss += module.get_weight_decay_loss() * self.opt.upconv_coeff
                else:
                    mask_weight_loss += module.get_weight_decay_loss()

        return mask_weight_loss

    def init_discriminator(self):

        if self.opt.pretrain_path is None or not os.path.exists(self.opt.pretrain_path):
            raise FileExistsError('The pretrain model path must be exist!!!')
        ckpt = torch.load(self.opt.pretrain_path, map_location=self.device)
        self.netD_A.load_state_dict(ckpt['D_A'])
        self.netD_B.load_state_dict(ckpt['D_B'])
        print('load pretrain discriminator weight done!!!')

    def stable_weight(self, model, bound):

        stepfunc_params = None
        last_bound = 1.0
        for module in model.modules():
            if isinstance(module, Mask):
                stepfunc_params = module.stepfunc_params
                last_bound = module.bound.data
                break
        state_dict = model.state_dict()

        mask_model_keys = ['model.1.', 'model.5.', 'model.9.']
        for i in range(13, 22, 1):
            mask_model_keys.append('model.%d.conv_block.1.conv.2.' % i)
            mask_model_keys.append('model.%d.conv_block.7.conv.2.' % i)
        mask_model_keys.append('model.22.')

        mask_weight_keys = ['model.3.mask_weight', 'model.7.mask_weight', 'model.11.mask_weight']
        for i in range(13, 22, 1):
            mask_weight_keys.append('model.%d.conv_block.3.mask_weight' % i)
            mask_weight_keys.append('model.%d.conv_block.9.mask_weight' % i)
        mask_weight_keys.append('model.24.mask_weight')
        if not self.opt.unmask_last_upconv:
            mask_model_keys.append('model.26.')
            mask_weight_keys.append('model.28.mask_weight')

        for i, mask_weight_key in enumerate(mask_weight_keys):

            mask_weight = state_dict[mask_weight_key]
            stable_weight_mask = (mask_weight > bound) & (mask_weight <= last_bound)

            for j in range(len(stable_weight_mask)):

                if stable_weight_mask[j]:
                    scale = (mask_weight[j] * stepfunc_params[3] + stepfunc_params[4]) * mask_weight[j] + stepfunc_params[5]
                    if i == len(mask_weight_keys)-1 or (i == len(mask_weight_keys) - 2 and not self.opt.unmask_last_upconv):
                        state_dict[mask_model_keys[i] + 'weight'][:, j, :, :] *= scale
                    else:
                        state_dict[mask_model_keys[i] + 'weight'][j] *= scale

                    state_dict[mask_model_keys[i] + 'bias'][j] *= scale

        model.load_state_dict(state_dict)

    def binary(self, model, boundary):

        for name, module in model.named_modules():

            if isinstance(module, Mask):

                one_index = module.mask_weight > boundary
                zero_idnex = module.mask_weight <= boundary

                module.mask_weight.data[one_index] = 1.0
                module.mask_weight.data[zero_idnex] = -1.0

    def get_cfg_residual_mask(self, state_dict, bound=0.0):

        prune_residual_keys = ['model.11.mask_weight'] + ['model.%d.conv_block.9.mask_weight' % i for i in
                                                          range(13, 22, 1)]

        residual_width = state_dict[prune_residual_keys[0]].size(0)
        residual_mask = [0] * residual_width
        for residual_key in prune_residual_keys:

            current_mask = state_dict[residual_key] > bound

            for i in range(len(current_mask)):
                if current_mask[i]:
                    residual_mask[i] += 1
        residual_mask = torch.FloatTensor(residual_mask) > int(self.opt.threshold)

        residual_cfg = sum(residual_mask)
        total_cfgs = []
        for k, v in state_dict.items():

            if str.endswith(k, '.mask_weight'):
                if k in prune_residual_keys:
                    total_cfgs.append(int(residual_cfg))
                else:
                    total_cfgs.append(int(sum(v > bound)))

        return total_cfgs, residual_mask

    def early_stop_mask(self):

        AtoB_bound = 1.0
        BtoA_bound = 1.0
        for module in self.netG_A.modules():
            if isinstance(module, Mask):
                AtoB_bound = module.bound.data
                break
        for module in self.netG_B.modules():
            if isinstance(module, Mask):
                BtoA_bound = module.bound.data
                break

        AtoB_cfgs, AtoB_residual_mask = self.get_cfg_residual_mask(self.netG_A.state_dict(), bound=-AtoB_bound)
        BtoA_cfgs, BtoA_residual_mask = self.get_cfg_residual_mask(self.netG_B.state_dict(), bound=-BtoA_bound)

        new_opt = copy.copy(self.opt)
        new_opt.mask = False
        pruned_model = MobileCycleGANModel(new_opt, cfg_AtoB=AtoB_cfgs, cfg_BtoA=BtoA_cfgs)

        input = torch.randn((1, self.opt.input_nc, self.opt.crop_size, self.opt.crop_size)).to(self.device)
        AtoB_macs, AtoB_params = profile(pruned_model.netG_A, inputs=(input, ), verbose=False)
        BtoA_macs, BtoA_params = profile(pruned_model.netG_B, inputs=(input, ), verbose=False)

        AtoB_macs = AtoB_macs / (1000 ** 3) # convert bit to GB
        # AtoB_params = AtoB_params / (1000 ** 2) # convert bit to MB
        BtoA_macs = BtoA_macs / (1000 ** 3)  # convert bit to GB
        # BtoA_params = BtoA_params / (1000 ** 2)  # convert bit to MB

        if AtoB_macs <= self.opt.AtoB_macs_threshold and not self.stop_AtoB_mask:
            self.stable_weight(self.netG_A, bound=-AtoB_bound)
            self.binary(self.netG_A, boundary=-AtoB_bound)
            self.stop_AtoB_mask = True

        if BtoA_macs <= self.opt.BtoA_macs_threshold and not self.stop_BtoA_mask:
            self.stable_weight(self.netG_B, bound=-BtoA_bound)
            self.binary(self.netG_B, boundary=-BtoA_bound)
            self.stop_BtoA_mask = True

    def prune(self, opt, logger):

        def inhert_weight(model, mask_model, residual_mask, bound=0.0, n_blocks=9, unmask_last_upconv=False):

            state_dict = model.state_dict()
            mask_state_dict = mask_model.state_dict()
            separable = False

            pruned_model_keys = ['model.1.', 'model.4.', 'model.7.']
            for i in range(10, 10+n_blocks, 1):
                pruned_model_keys.append('model.%d.conv_block.1.' % i)
                pruned_model_keys.append('model.%d.conv_block.6.' % i)
            pruned_model_keys.append('model.%d.' % (19 - (9-n_blocks)))
            pruned_model_keys.append('model.%d.' % (22 - (9-n_blocks)))
            pruned_model_keys.append('model.%d.' % (26 - (9-n_blocks)))

            mask_model_keys = ['model.1.', 'model.5.', 'model.9.']
            for i in range(13, 13+9, 1):
                mask_model_keys.append('model.%d.conv_block.1.' % i)
                mask_model_keys.append('model.%d.conv_block.7.' % i)
            mask_model_keys.append('model.22.')
            mask_model_keys.append('model.26.')
            if self.opt.unmask_last_upconv:
                mask_model_keys.append('model.30.')
            else:
                mask_model_keys.append('model.31.')

            mask_weight_keys = ['model.3.mask_weight', 'model.7.mask_weight', 'model.11.mask_weight']
            for i in range(13, 13+9, 1):
                mask_weight_keys.append('model.%d.conv_block.3.mask_weight' % i)
                mask_weight_keys.append('model.%d.conv_block.9.mask_weight' % i)
            mask_weight_keys.append('model.24.mask_weight')
            mask_weight_keys.append('model.28.mask_weight')

            last_mask = None
            pass_flag = False
            pruned_model_keys_index = 0
            for i, mask_model_key in enumerate(mask_model_keys):

                new_filter_index = 0
                new_channel_index = 0

                mask_weight_key = mask_weight_keys[i % len(mask_weight_keys)] # last conv has not mask_weight
                if mask_weight_key in mask_state_dict.keys():
                    current_mask = mask_state_dict[mask_weight_key] > bound
                else:
                    current_mask = last_mask

                if pass_flag:  # Second layer in the block can be remove
                    pass_flag = False
                    continue
                if int(sum(current_mask)) == 0:  # First layer in the block can be remove
                    print('pass', mask_model_key)
                    pass_flag = True
                    continue

                pruned_model_key = pruned_model_keys[pruned_model_keys_index]
                pruned_model_keys_index += 1

                if i == 0: # only prune filter
                    print('Pruning1: ', mask_model_key)
                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            state_dict[pruned_model_key+'weight'][new_filter_index, :, :, :] = \
                                mask_state_dict[mask_model_key+'weight'][j, :, :, :]
                            state_dict[pruned_model_key+'bias'][new_filter_index] = \
                                mask_state_dict[mask_model_key+'bias'][j]
                            new_filter_index += 1

                elif i == len(mask_model_keys) - 1: # last conv only prune channel
                    print('Pruning2: ', mask_model_key)

                    for j in range(len(last_mask)):
                        if last_mask[j]:
                            state_dict[pruned_model_key+'weight'][:, new_channel_index, :, :] = \
                                mask_state_dict[mask_model_key+'weight'][:, j, :, :]
                            new_channel_index += 1
                    state_dict[pruned_model_key+'bias'] = mask_state_dict[mask_model_key+'bias']

                elif i == len(mask_model_keys) - 2 or i == len(mask_model_keys) - 3: # upconv prune
                    print('Pruning3: ', mask_model_key)

                    if unmask_last_upconv and i == len(mask_model_keys) - 2:
                        current_mask = [True for _ in range(mask_state_dict[mask_model_key+'bias'].size(0))]

                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            new_channel_index = 0
                            for k in range(len(last_mask)):
                                if last_mask[k]:
                                    state_dict[pruned_model_key+'weight'][new_channel_index, new_filter_index, :, :] = \
                                        mask_state_dict[mask_model_key+'weight'][k, j, :, :]
                                    new_channel_index += 1
                            state_dict[pruned_model_key+'bias'][new_filter_index] = mask_state_dict[mask_model_key+'bias'][j]
                            new_filter_index += 1

                else:
                    print('Pruning4: ', mask_model_key)

                    if i % 2 == 0: # prune last conv in block
                        zero_mask = current_mask
                        current_mask = residual_mask
                    else:
                        zero_mask = [True for _ in range(len(current_mask))]
                    if separable: # separable conv

                        for j in range(len(last_mask)):
                            if last_mask[j]:
                                state_dict[pruned_model_key + 'conv.0.weight'][new_channel_index, :, :, :] = \
                                    mask_state_dict[mask_model_key + 'conv.0.weight'][j, :, :, :]
                                state_dict[pruned_model_key + 'conv.0.bias'][new_channel_index] = \
                                    mask_state_dict[mask_model_key + 'conv.0.bias'][j]
                                new_filter_index = 0
                                for k in range(len(current_mask)):
                                    if current_mask[k]:
                                        state_dict[pruned_model_key + 'conv.2.weight'][new_filter_index, new_channel_index, :, :] = \
                                            mask_state_dict[mask_model_key + 'conv.2.weight'][k, j, :, :] * 1.0 if zero_mask[k] else 0.0
                                        state_dict[pruned_model_key + 'conv.2.bias'][new_filter_index] = \
                                            mask_state_dict[mask_model_key + 'conv.2.bias'][k] * 1.0 if zero_mask[k] else 0.0
                                        new_filter_index += 1
                                new_channel_index += 1
                    else: # first conv in residul block is normal conv

                        for j in range(len(current_mask)):
                            if current_mask[j]:
                                new_channel_index = 0
                                for k in range(len(last_mask)):
                                    if last_mask[k]:
                                        state_dict[pruned_model_key + 'weight'][new_filter_index, new_channel_index, :, :] = \
                                            mask_state_dict[mask_model_key + 'weight'][j, k, :, :] * 1.0 if zero_mask[j] else 0.0
                                        new_channel_index += 1
                                state_dict[pruned_model_key + 'bias'][new_filter_index] = \
                                    mask_state_dict[mask_model_key + 'bias'][j] * 1.0 if zero_mask[j] else 0.0
                                new_filter_index += 1
                        if i % 2 == 0:
                            separable = True

                last_mask = current_mask

            model.load_state_dict(state_dict)

        AtoB_fid, BtoA_fid = self.load_models(opt.load_path)
        logger.info('After Training. AtoB FID: %.2f\tBtoA FID: %.2f' % (AtoB_fid, BtoA_fid))

        AtoB_cfgs, AtoB_residual_mask = self.get_cfg_residual_mask(self.netG_A.state_dict())
        BtoA_cfgs, BtoA_residual_mask = self.get_cfg_residual_mask(self.netG_B.state_dict())

        logger.info(AtoB_cfgs)
        logger.info(BtoA_cfgs)

        new_opt = copy.copy(self.opt)
        new_opt.mask = False
        pruned_model = MobileCycleGANModel(new_opt, cfg_AtoB=AtoB_cfgs, cfg_BtoA=BtoA_cfgs)
        inhert_weight(pruned_model.netG_A, self.netG_A, AtoB_residual_mask, n_blocks=9-AtoB_cfgs.count(0), unmask_last_upconv=opt.unmask_last_upconv)
        inhert_weight(pruned_model.netG_B, self.netG_B, BtoA_residual_mask, n_blocks=9-BtoA_cfgs.count(0), unmask_last_upconv=opt.unmask_last_upconv)
        ckpt = torch.load(opt.load_path, map_location=self.device)
        pruned_model.netD_A.load_state_dict(ckpt['D_A'])
        pruned_model.netD_B.load_state_dict(ckpt['D_B'])

        logger.info('Prune done!!!')
        return pruned_model

# class opts():
#     mask_loss_type = 'relu'
#     upconv_solo = True,
#     upconv_bound = True,
#     upconv_relu = False
# moblie_model = MobileResnetGenerator(input_nc=3, output_nc=3, ngf=64, opt=opts())
# print(moblie_model)
#
# for k, v in moblie_model.state_dict().items():
#     print(k, v.size())
#
# input = torch.randn((1, 3, 256, 256))
# print(moblie_model(input).size())
#
# macs, params = profile(moblie_model, (input, ))
# print(macs, params)