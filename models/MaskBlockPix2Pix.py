import torch
import torch.nn as nn

from models.GANLoss import GANLoss
import utils.util as util
from models.MaskLayer import Mask
from models.BlockPix2Pix import BlockPix2PixModel

import functools
import os
import copy
import math
from collections import OrderedDict

class MaskResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, opt=None):
        super(MaskResnetBlock, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):

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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       Mask(dim, mask_loss_type=self.opt.mask_loss_type),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       Mask(dim, mask_loss_type=self.opt.mask_loss_type),
                       ]

        return nn.Sequential(*conv_block)

    def forward(self, x):

        out = x + self.conv_block(x)
        return out

class MaskResnetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect', opt=None):
        super(MaskResnetGenerator, self).__init__()
        assert(n_blocks >= 0)
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if self.opt.upconv_relu:
            upconv_mask_loss_type = 'uprelu'
        elif self.opt.upconv_bound:
            upconv_mask_loss_type = 'bound'
        else:
            upconv_mask_loss_type = 'relu'

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 Mask(ngf, mask_loss_type=upconv_mask_loss_type),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      Mask(ngf * mult * 2, mask_loss_type=self.opt.mask_loss_type),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):

            model += [MaskResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, opt=self.opt)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if i == n_downsampling - 1 and self.opt.unmask_last_upconv:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          Mask(int(ngf * mult / 2), mask_loss_type=upconv_mask_loss_type),
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
            group_mask_weight_names.append('model.%d.conv_block.8' % i)

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
            group_mask_weight_names.append('model.%d.conv_block.8' % i)

        residual_mask = [True for _ in range(256)]

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
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
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

class MaskBlockPix2PixModel(nn.Module):

    def __init__(self, opt):
        super(MaskBlockPix2PixModel, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'mask_weight']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.netG = MaskResnetGenerator(input_nc=3, output_nc=3, ngf=opt.ngf, opt=self.opt)
        self.netD = NLayerDiscriminator(input_nc=3+3, ndf=128)
        self.init_net()

        self.group_mask_weight_names = []
        self.group_mask_weight_names.append('model.11')
        for i in range(13, 22, 1):
            self.group_mask_weight_names.append('model.%d.conv_block.8' % i)

        self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
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
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

        if self.opt.lambda_distill > 0: # extract teacher attention
            self.teacher_model.netG(self.real_A)  # G(A)
            self.teacher_model.netD(self.fake_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Mask weight decay loss
        self.loss_mask_weight = self.get_mask_weight_loss(self.netG) * self.opt.mask_weight_decay
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_mask_weight
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def update_learning_rate(self, epoch):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        self.netG.update_sparsity_factor()
        self.update_masklayer(epoch)

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
            'G': self.netG.state_dict(),
            'D': self.netD.state_dict(),
            'epoch': epoch,
            'fid': fid
        }
        if isbest:
            torch.save(ckpt, os.path.join(save_dir, 'model_best_%s.pth' % direction))
        else:
            torch.save(ckpt, os.path.join(save_dir, 'model_%d.pth' % epoch))

    def load_models(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device)
        self.netG.load_state_dict(ckpt['G'])
        self.netD.load_state_dict(ckpt['D'])

        print('loading the model from %s' % load_path)
        return ckpt['fid'], float('inf')

    def init_net(self):
        self.netG.to(self.device)
        self.netD.to(self.device)

        util.init_weights(self.netG, init_type='normal', init_gain=0.02)
        util.init_weights(self.netD, init_type='normal', init_gain=0.02)

    def model_train(self):
        self.netG.train()
        self.netD.train()

    def model_eval(self):
        self.netG.eval()
        self.netD.eval()

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

    def update_masklayer(self, epoch):

        update_bound_epochs_count = (self.opt.n_epochs + self.opt.n_epochs_decay) * 0.75

        if epoch > update_bound_epochs_count:
            bound = 0.0
        else:
            if self.opt.update_bound_rule == 'cube':
                bound = 1 - math.pow(float(epoch) / update_bound_epochs_count, 1 / 3)
            else:
                bound = 1 - math.pow(float(epoch) / update_bound_epochs_count, 1 / 2)
            if bound < 0:
                bound = 0.0
        print('Bound: %.3f' % bound)

        self.stable_weight(self.netG, bound=bound)
        self.netG.update_masklayer(bound)

    def print_sparsity_info(self, logger):
        logger.info('netG')
        self.netG.print_sparse_info(logger)

    def get_mask_weight_loss(self, G):
        mask_weight_loss = 0.0
        for name, module in G.named_modules():

            if isinstance(module, Mask):
                if self.opt.lambda_update_coeff > 0 and name in self.group_mask_weight_names:
                    mask_weight_loss += module.get_block_decay_loss(G.block_sparsity_coeff)
                elif name == 'model.28' or name == 'model.24' or name == 'model.3':
                    mask_weight_loss += module.get_weight_decay_loss() * self.opt.upconv_coeff
                else:
                    mask_weight_loss += module.get_weight_decay_loss()

        return mask_weight_loss

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
            mask_model_keys.append('model.%d.conv_block.1.' % i)
            mask_model_keys.append('model.%d.conv_block.6.' % i)
        mask_model_keys.append('model.22.')

        mask_weight_keys = ['model.3.mask_weight', 'model.7.mask_weight', 'model.11.mask_weight']
        for i in range(13, 22, 1):
            mask_weight_keys.append('model.%d.conv_block.3.mask_weight' % i)
            mask_weight_keys.append('model.%d.conv_block.8.mask_weight' % i)
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
                        state_dict[mask_model_keys[i] + 'weight'][j] *= scale
                    else:
                        state_dict[mask_model_keys[i] + 'weight'][j] *= scale

        model.load_state_dict(state_dict)

    def prune(self, opt, logger):

        def get_cfg_residual_mask(state_dict, bound=0.0):

            prune_residual_keys = ['model.11.mask_weight'] + ['model.%d.conv_block.8.mask_weight' % i for i in
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


        def inhert_weight(model, mask_model, residual_mask, bound=0.0, n_blocks=9, unmask_last_upconv=False):

            state_dict = model.state_dict()
            mask_state_dict = mask_model.state_dict()

            pruned_model_keys = ['model.1.', 'model.4.', 'model.7.']
            for i in range(10, 10+n_blocks, 1):
                pruned_model_keys.append('model.%d.conv_block.1.' % i)
                pruned_model_keys.append('model.%d.conv_block.5.' % i)
            pruned_model_keys.append('model.%d.' % (19 - (9-n_blocks)))
            pruned_model_keys.append('model.%d.' % (22 - (9-n_blocks)))
            pruned_model_keys.append('model.%d.' % (26 - (9-n_blocks)))

            mask_model_keys = ['model.1.', 'model.5.', 'model.9.']
            for i in range(13, 13+9, 1):
                mask_model_keys.append('model.%d.conv_block.1.' % i)
                mask_model_keys.append('model.%d.conv_block.6.' % i)
            mask_model_keys.append('model.22.')
            mask_model_keys.append('model.26.')
            if self.opt.unmask_last_upconv:
                mask_model_keys.append('model.30.')
            else:
                mask_model_keys.append('model.31.')

            mask_weight_keys = ['model.3.mask_weight', 'model.7.mask_weight', 'model.11.mask_weight']
            for i in range(13, 13+9, 1):
                mask_weight_keys.append('model.%d.conv_block.3.mask_weight' % i)
                mask_weight_keys.append('model.%d.conv_block.8.mask_weight' % i)
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
                        current_mask = [True for _ in range(mask_state_dict[mask_model_key+'weight'].size(1))]

                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            new_channel_index = 0
                            for k in range(len(last_mask)):
                                if last_mask[k]:
                                    state_dict[pruned_model_key+'weight'][new_channel_index, new_filter_index, :, :] = \
                                        mask_state_dict[mask_model_key+'weight'][k, j, :, :]
                                    new_channel_index += 1
                            new_filter_index += 1

                else:
                    print('Pruning4: ', mask_model_key)

                    if i % 2 == 0: # prune last conv in block
                        zero_mask = current_mask
                        current_mask = residual_mask
                    else:
                        zero_mask = [True for _ in range(len(current_mask))]
                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            new_channel_index = 0
                            for k in range(len(last_mask)):
                                if last_mask[k]:
                                    state_dict[pruned_model_key+'weight'][new_filter_index, new_channel_index, :, :] = \
                                        mask_state_dict[mask_model_key+'weight'][j, k, :, :] * 1.0 if zero_mask[j] else 0.0
                                    new_channel_index += 1

                            new_filter_index += 1

                last_mask = current_mask

            model.load_state_dict(state_dict)

        AtoB_fid, _ = self.load_models(opt.load_path)
        logger.info('After Training. FID: %.2f' % AtoB_fid)

        cfgs, residual_mask = get_cfg_residual_mask(self.netG.state_dict())

        logger.info(cfgs)

        new_opt = copy.copy(self.opt)
        new_opt.mask = False
        pruned_model = BlockPix2PixModel(new_opt, cfg=cfgs)

        inhert_weight(pruned_model.netG, self.netG, residual_mask, n_blocks=9-cfgs.count(0), unmask_last_upconv=opt.unmask_last_upconv)
        ckpt = torch.load(opt.load_path, map_location=self.device)
        pruned_model.netD.load_state_dict(ckpt['D'])

        logger.info('Prune done!!!')
        return pruned_model