import torch
import torch.nn as nn

from models.GANLoss import GANLoss
import utils.util as util

import functools
import os
import copy
from collections import OrderedDict

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

class MobileResnetBlock(nn.Module):
    def __init__(self, layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate, use_bias, opt=None):
        super(MobileResnetBlock, self).__init__()
        self.opt = opt
        self.conv_block = self.build_conv_block(layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, layer1_input_dim, layer1_output_dim, layer2_output_dim, padding_type, norm_layer, dropout_rate, use_bias):
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
            SeparableConv2d(in_channels=layer1_input_dim, out_channels=layer1_output_dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(layer1_output_dim),
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
            SeparableConv2d(in_channels=layer1_output_dim, out_channels=layer2_output_dim,
                            kernel_size=3, padding=p, stride=1),
            norm_layer(layer2_output_dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MobileResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 dropout_rate=0, n_blocks=9, padding_type='reflect', opt=None, cfg=None):
        assert (n_blocks >= 0)
        super(MobileResnetGenerator, self).__init__()
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        cfg_index = 0
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf if cfg is None else cfg[cfg_index], kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            input_channel_num = ngf * mult if cfg is None else cfg[cfg_index - 1]
            output_channel_num = ngf * mult * 2 if cfg is None else cfg[cfg_index]
            model += [nn.Conv2d(input_channel_num, output_channel_num, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_channel_num),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):
            block_layer1_input_channel_num = ngf * mult if cfg is None else cfg[cfg_index - 1]
            block_layer1_output_channel_num = ngf * mult if cfg is None else cfg[cfg_index]
            cfg_index += 1
            block_layer2_output_channel_num = ngf * mult if cfg is None else cfg[cfg_index]
            cfg_index += 1
            if block_layer1_output_channel_num == 0:
                continue
            model += [MobileResnetBlock(block_layer1_input_channel_num,
                                    block_layer1_output_channel_num,
                                    block_layer2_output_channel_num,
                                    padding_type=padding_type, norm_layer=norm_layer,
                                    dropout_rate=dropout_rate, use_bias=use_bias, opt=self.opt)]

        output_channel_num = ngf
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            input_channel_num = ngf * mult if cfg is None or cfg_index == 0 else cfg[cfg_index - 1]
            output_channel_num = int(ngf * mult / 2) if cfg is None else cfg[cfg_index]
            cfg_index += 1
            model += [nn.ConvTranspose2d(input_channel_num, output_channel_num,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(output_channel_num, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

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

class MobilePix2PixModel(nn.Module):

    def __init__(self, opt, cfg=None):
        super(MobilePix2PixModel, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.netG = MobileResnetGenerator(ngf=self.opt.ngf, opt=self.opt, cfg=cfg)
        self.netD = NLayerDiscriminator(input_nc=3+3, ndf=128)
        self.init_net()

        if self.opt.lambda_distill > 0:
            print('init distill')
            self.init_distill()

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
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
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

    def init_distill(self):
        if self.opt.pretrain_path is None or not os.path.exists(self.opt.pretrain_path):
            raise FileExistsError('The pretrain model path must be exist!!!')
        new_opt = copy.copy(self.opt)
        new_opt.lambda_distill = 0.0
        self.teacher_model = MobilePix2PixModel(new_opt)
        self.teacher_model.load_models(self.opt.pretrain_path)

        self.loss_names.append('attention_distill')

        self.total_feature_out_teacher = {}
        self.total_feature_out_student = {}
        self.total_feature_out_D_teacher = {}

        self.teacher_extract_G_layers = ['model.9', 'model.12', 'model.15', 'model.18']
        self.teacher_extract_D_layers = ['model.4', 'model.10']
        self.student_extract_G_layers = ['model.9', 'model.12', 'model.15', 'model.18']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output

            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self.teacher_model.netG, self.total_feature_out_teacher, self.teacher_extract_G_layers)
        add_hook(self.teacher_model.netD, self.total_feature_out_D_teacher, self.teacher_extract_D_layers)
        add_hook(self.netG, self.total_feature_out_student, self.student_extract_G_layers)

    def distill_loss(self):

        total_attention_teacher = [f.pow(2).mean(1, keepdim=True) for f in
                                        self.total_feature_out_AtoB_teacher.values()]
        total_attention_D_teacher = [f.pow(2).mean(1, keepdim=True) for f in
                                      self.total_feature_out_DA_teacher.values()]
        total_attention_student = [f.pow(2).mean(1, keepdim=True) for f in
                                        self.total_feature_out_AtoB_student.values()]

        # interpolate attention map from 31*31 to 64*64
        total_attention_D_teacher[1] = util.attention_interpolate(total_attention_D_teacher[1])

        total_mixup_attention = []

        for i in range(len(total_attention_teacher)):
            total_mixup_attention.append(util.mixup_attention(
                [total_attention_teacher[i], total_attention_D_teacher[0], total_attention_D_teacher[1]],
                [0.5, 0.25, 0.25]))

        total_distill_loss = 0.0
        for i in range(len(total_attention_teacher)):
            total_distill_loss += util.attention_loss(total_mixup_attention[i], total_attention_student[i],
                                                      normalize=self.opt.attention_normal)
        return total_distill_loss