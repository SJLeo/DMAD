import copy
import math

import torch
import torch.nn as nn

from models.GANLoss import GANLoss
from models.MaskLayer import Mask
from models.Pix2Pix import Pix2PixModel
import utils.util as util

import functools
import os
from scipy.interpolate import lagrange
from collections import OrderedDict

class MaskUnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, opt=None):
        super(MaskUnetSkipConnectionBlock, self).__init__()
        self.opt = opt
        self.outermost = outermost
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        downmask = Mask(inner_nc, mask_loss_type=self.opt.mask_loss_type)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        upmask = Mask(outer_nc, mask_loss_type=self.opt.mask_loss_type)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downmask]
            up = [uprelu, upconv, upnorm, upmask]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm, downmask]
            up = [uprelu, upconv, upnorm, upmask]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class MaskUnetGenertor(nn.Module):

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, opt=None):
        super(MaskUnetGenertor, self).__init__()
        self.opt = opt
        unet_block = MaskUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True, opt=self.opt)
        for i in range(num_downs - 5):
            unet_block = MaskUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout, opt=self.opt)

        unet_block = MaskUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, opt=self.opt)
        unet_block = MaskUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, opt=self.opt)
        unet_block = MaskUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, opt=self.opt)
        self.model = MaskUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, opt=self.opt)

    def update_masklayer(self, bound):
        for module in self.modules():
            if isinstance(module, Mask):
                module.update(bound)

    def print_sparse_info(self, logger):

        for name, module in self.named_modules():
            if isinstance(module, Mask):

                mask = module.get_current_mask()

                logger.info('%s sparsity ratio: %.2f\tone ratio: %.2f\t'
                      'total ratio: %.2f' % (name, float(sum(mask == 0.0)) / mask.numel(),
                                              float(sum(mask == 1.0)) / mask.numel(),
                                             (float(sum(mask == 0.0)) + float(sum(mask == 1.0))) / mask.numel()))

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

class MaskPix2PixModel(nn.Module):

    def __init__(self, opt):
        super(MaskPix2PixModel, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'mask_weight']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.netG = MaskUnetGenertor(input_nc=3, output_nc=3, num_downs=8, ngf=opt.ngf, use_dropout=not opt.no_dropout, opt=self.opt)
        self.netD = NLayerDiscriminator(input_nc=3+3, ndf=128)
        self.init_net()

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
        # mask weight decay
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

        if self.opt.isTrain:
            self.update_masklayer(self.opt.epoch_count-1)

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

        mask_model_keys = []
        mask_model_bn_keys = []
        mask_weight_keys = []
        for i in range(7):
            mask_model_key = 'model.model.1.'
            for j in range(i):
                mask_model_key += 'model.4.'
            mask_model_bn_keys.append(mask_model_key + 'model.2.')
            if i != 6:
                mask_weight_keys.append(mask_model_key + 'model.3.mask_weight')
            else:
                mask_weight_keys.append(mask_model_key + 'model.2.mask_weight')

        mask_model_bn_keys.append('model.model.1.model.4.model.4.model.4.model.4.model.4.model.4.model.5.')
        mask_weight_keys.append('model.model.1.model.4.model.4.model.4.model.4.model.4.model.4.model.6.mask_weight')
        for i in range(5, -1, -1):
            mask_model_key = 'model.model.1.'
            for j in range(i):
                mask_model_key += 'model.4.'
            mask_model_bn_keys.append(mask_model_key + 'model.7.')
            mask_weight_keys.append(mask_model_key + 'model.8.mask_weight')

        for i, mask_weight_key in enumerate(mask_weight_keys):

            mask_weight = state_dict[mask_weight_key]
            stable_weight_mask = (mask_weight > bound) & (mask_weight <= last_bound)

            for j in range(len(stable_weight_mask)):

                if stable_weight_mask[j]:
                    scale = (mask_weight[j] * stepfunc_params[3] + stepfunc_params[4]) * mask_weight[j] + stepfunc_params[5]
                    state_dict[mask_model_bn_keys[i] + 'weight'][j] *= scale
                    state_dict[mask_model_bn_keys[i] + 'bias'][j] *= scale

        model.load_state_dict(state_dict)

    def prune(self, opt, logger):

        def get_cfg(state_dict, bound=0.0):

            total_filter_cfgs = []
            total_channel_cfgs = []
            for k, v in state_dict.items():

                if str.endswith(k, '.mask_weight'):

                    if len(total_filter_cfgs) == 0: # first conv not prune
                        total_channel_cfgs.append(self.opt.ngf)
                    elif len(total_filter_cfgs) < 8: # half conv's channel num only depend on last conv
                        total_channel_cfgs.append(total_filter_cfgs[-1])
                    elif len(total_filter_cfgs) < 14:
                        total_channel_cfgs.append(total_filter_cfgs[13-len(total_filter_cfgs)] + total_filter_cfgs[-1])

                    total_filter_cfgs.append(int(sum(v > bound)))

            total_channel_cfgs.append(self.opt.ngf + total_filter_cfgs[-1]) # outermost
            return total_filter_cfgs, total_channel_cfgs

        def inhert_weight(model, mask_model, bound=0.0):

            state_dict = model.state_dict()
            mask_state_dict = mask_model.state_dict()

            state_dict['model.model.0.weight'] = mask_state_dict['model.model.0.weight']
            state_dict['model.model.3.bias'] = mask_state_dict['model.model.3.bias']

            pruned_model_keys = []
            pruned_model_bn_keys = []
            for i in range(7):
                pruned_model_key = 'model.model.1.'
                for j in range(i):
                    pruned_model_key += 'model.3.'
                pruned_model_bn_keys.append(pruned_model_key + 'model.2.')
                pruned_model_key += 'model.1.weight'
                pruned_model_keys.append(pruned_model_key)

            pruned_model_keys.append('model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.3.weight')
            pruned_model_bn_keys.append('model.model.1.model.3.model.3.model.3.model.3.model.3.model.3.model.4.')
            for i in range(5, -1, -1):
                pruned_model_key = 'model.model.1.'
                for j in range(i):
                    pruned_model_key += 'model.3.'
                pruned_model_bn_keys.append(pruned_model_key + 'model.6.')
                pruned_model_key += 'model.5.weight'
                pruned_model_keys.append(pruned_model_key)
            pruned_model_keys.append('model.model.3.weight')

            mask_model_keys = []
            mask_model_bn_keys = []
            mask_weight_keys = []
            for i in range(7):
                mask_model_key = 'model.model.1.'
                for j in range(i):
                    mask_model_key += 'model.4.'
                mask_model_bn_keys.append(mask_model_key + 'model.2.')
                if i != 6:
                    mask_weight_keys.append(mask_model_key + 'model.3.mask_weight')
                else:
                    mask_weight_keys.append(mask_model_key + 'model.2.mask_weight')
                mask_model_key += 'model.1.weight'
                mask_model_keys.append(mask_model_key)

            mask_model_keys.append('model.model.1.model.4.model.4.model.4.model.4.model.4.model.4.model.4.weight')
            mask_model_bn_keys.append('model.model.1.model.4.model.4.model.4.model.4.model.4.model.4.model.5.')
            mask_weight_keys.append('model.model.1.model.4.model.4.model.4.model.4.model.4.model.4.model.6.mask_weight')
            for i in range(5, -1, -1):
                mask_model_key = 'model.model.1.'
                for j in range(i):
                    mask_model_key += 'model.4.'
                mask_model_bn_keys.append(mask_model_key + 'model.7.')
                mask_weight_keys.append(mask_model_key + 'model.8.mask_weight')
                mask_model_key += 'model.6.weight'
                mask_model_keys.append(mask_model_key)
            mask_model_keys.append('model.model.3.weight')

            last_mask = None

            has_pruned_keys = []

            for i, pruned_model_key in enumerate(pruned_model_keys):

                pruned_model_bn_key = pruned_model_bn_keys[i % len(pruned_model_bn_keys)] # last conv has not bn
                mask_model_key = mask_model_keys[i]
                mask_model_bn_key = mask_model_bn_keys[i % len(mask_model_bn_keys)] # last conv has not bn
                mask_weight_key = mask_weight_keys[i % len(mask_weight_keys)] # last conv has not mask_layer
                current_mask = mask_state_dict[mask_weight_key] > bound

                new_filter_index = 0
                new_channel_index = 0

                if i == 0: # only prune filter
                    print('Pruning1: ', pruned_model_key)

                    has_pruned_keys.append(pruned_model_key)
                    has_pruned_keys.append(pruned_model_bn_key+'weight')
                    has_pruned_keys.append(pruned_model_bn_key+'bias')
                    has_pruned_keys.append(pruned_model_bn_key+'running_mean')
                    has_pruned_keys.append(pruned_model_bn_key+'running_var')

                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            state_dict[pruned_model_key][new_filter_index] = mask_state_dict[mask_model_key][j]
                            state_dict[pruned_model_bn_key+'weight'][new_filter_index] = mask_state_dict[mask_model_bn_key+'weight'][j]
                            state_dict[pruned_model_bn_key+'bias'][new_filter_index] = mask_state_dict[mask_model_bn_key+'bias'][j]
                            state_dict[pruned_model_bn_key+'running_mean'][new_filter_index] = mask_state_dict[mask_model_bn_key+'running_mean'][j]
                            state_dict[pruned_model_bn_key+'running_var'][new_filter_index] = mask_state_dict[mask_model_bn_key+'running_var'][j]
                            new_filter_index += 1
                elif i == len(pruned_model_keys) - 1: # only prune channel
                    print('Pruning2: ', pruned_model_key)

                    for j in range(self.opt.ngf):
                        state_dict[pruned_model_key][j, :, :, :] = mask_state_dict[mask_model_key][j, :, :, :]
                    for j in range(len(last_mask)):
                        if last_mask[j]:
                            state_dict[pruned_model_key][self.opt.ngf + new_channel_index, :, :, :] = \
                                mask_state_dict[mask_model_key][self.opt.ngf + j, :, :, :]
                            new_channel_index += 1
                elif i <= 6: # downconv
                    print('Pruning3: ', pruned_model_key)
                    has_pruned_keys.append(pruned_model_key)
                    has_pruned_keys.append(pruned_model_bn_key + 'weight')
                    has_pruned_keys.append(pruned_model_bn_key + 'bias')
                    has_pruned_keys.append(pruned_model_bn_key + 'running_mean')
                    has_pruned_keys.append(pruned_model_bn_key + 'running_var')

                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            new_channel_index = 0
                            for k in range(len(last_mask)):
                                if last_mask[k]:
                                    state_dict[pruned_model_key][new_filter_index, new_channel_index, :, :] = \
                                        mask_state_dict[mask_model_key][j, k, :, :]
                                    new_channel_index += 1
                            if (pruned_model_bn_key + 'weight') in state_dict.keys(): # innermost conv has not bn
                                state_dict[pruned_model_bn_key + 'weight'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'weight'][j]
                                state_dict[pruned_model_bn_key + 'bias'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'bias'][j]
                                state_dict[pruned_model_bn_key + 'running_mean'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'running_mean'][j]
                                state_dict[pruned_model_bn_key + 'running_var'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'running_var'][j]

                            new_filter_index += 1
                elif i == 7: # innermost upconv
                    print('Pruning4: ', pruned_model_key)
                    has_pruned_keys.append(pruned_model_key)
                    has_pruned_keys.append(pruned_model_bn_key + 'weight')
                    has_pruned_keys.append(pruned_model_bn_key + 'bias')
                    has_pruned_keys.append(pruned_model_bn_key + 'running_mean')
                    has_pruned_keys.append(pruned_model_bn_key + 'running_var')

                    for j in range(len(current_mask)):
                        if current_mask[j]:
                            new_channel_index = 0
                            for k in range(len(last_mask)):
                                if last_mask[k]:
                                    state_dict[pruned_model_key][new_channel_index, new_filter_index, :, :] = \
                                        mask_state_dict[mask_model_key][k, j, :, :]
                                    new_channel_index += 1
                            state_dict[pruned_model_bn_key + 'weight'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'weight'][j]
                            state_dict[pruned_model_bn_key + 'bias'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'bias'][j]
                            state_dict[pruned_model_bn_key + 'running_mean'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'running_mean'][j]
                            state_dict[pruned_model_bn_key + 'running_var'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'running_var'][j]
                            new_filter_index += 1
                else: # other upconv
                    print('Pruning5: ', pruned_model_key)
                    has_pruned_keys.append(pruned_model_key)
                    has_pruned_keys.append(pruned_model_bn_key + 'weight')
                    has_pruned_keys.append(pruned_model_bn_key + 'bias')
                    has_pruned_keys.append(pruned_model_bn_key + 'running_mean')
                    has_pruned_keys.append(pruned_model_bn_key + 'running_var')
                    mapping_conv_mask = mask_state_dict[mask_weight_keys[13-i]] > bound

                    for j in range(len(current_mask)):

                        if current_mask[j]:
                            new_channel_index = 0
                            for k in range(len(mapping_conv_mask)):
                                if mapping_conv_mask[k]:
                                    state_dict[pruned_model_key][new_channel_index, new_filter_index, :, :] = \
                                        mask_state_dict[mask_model_key][k, j, :, :]
                                    new_channel_index += 1
                            for k in range(len(last_mask)):
                                if last_mask[k]:
                                    state_dict[pruned_model_key][new_channel_index, new_filter_index, :, :] = \
                                        mask_state_dict[mask_model_key][len(mapping_conv_mask) + k, j, :, :]
                                    new_channel_index += 1
                            state_dict[pruned_model_bn_key + 'weight'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'weight'][j]
                            state_dict[pruned_model_bn_key + 'bias'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'bias'][j]
                            state_dict[pruned_model_bn_key + 'running_mean'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'running_mean'][j]
                            state_dict[pruned_model_bn_key + 'running_var'][new_filter_index] = mask_state_dict[mask_model_bn_key + 'running_var'][j]
                            new_filter_index += 1

                last_mask = current_mask

            # for k, v in state_dict.items():
            #     if k not in has_pruned_keys:
            #         print(k)
            model.load_state_dict(state_dict)

        fid, _ = self.load_models(opt.load_path)
        logger.info('After Training. FID: %.2f' % fid)

        filter_cfgs, channel_cfgs = get_cfg(self.netG.state_dict())

        logger.info(filter_cfgs)
        logger.info(channel_cfgs)

        new_opt = copy.copy(self.opt)
        new_opt.mask = False
        pruned_model = Pix2PixModel(new_opt, filter_cfgs=filter_cfgs, channel_cfgs=channel_cfgs)

        inhert_weight(pruned_model.netG, self.netG, bound=0.0)

        logger.info('Prune done!!!')
        return pruned_model