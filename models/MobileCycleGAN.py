import torch
import torch.nn as nn

import utils.util as util
from utils.image_pool import ImagePool
from models.GANLoss import GANLoss

import itertools
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
        cfg_index += 1

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            input_channel_num = ngf * mult if cfg is None else cfg[cfg_index - 1]
            output_channel_num = ngf * mult * 2 if cfg is None else cfg[cfg_index]
            cfg_index += 1
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

class MobileCycleGANModel(nn.Module):

    def __init__(self, opt, cfg_AtoB=None, cfg_BtoA=None):
        super(MobileCycleGANModel, self).__init__()
        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.cfg_AtoB = cfg_AtoB
        self.cfg_BtoA = cfg_BtoA
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'idt_B']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'idt_A']
        self.visual_names = visual_names_A + visual_names_B

        self.netG_A = MobileResnetGenerator(opt=self.opt, cfg=cfg_AtoB)
        self.netG_B = MobileResnetGenerator(opt=self.opt, cfg=cfg_BtoA)

        self.netD_A = NLayerDiscriminator()
        self.netD_B = NLayerDiscriminator()
        self.init_net()

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        self.teacher_model = None
        if self.opt.lambda_attention_distill > 0:
            print('init attention distill')
            self.init_attention_distill()
        if self.opt.lambda_discriminator_distill > 0:
            print('init discriminator distill')
            self.init_discriminator_distill()

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

        if self.opt.lambda_attention_distill > 0 or self.opt.lambda_discriminator_distill > 0:

            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)

            teacher_fake_B = self.teacher_model.netG_A(self.real_A)  # G_A(A)
            teacher_fake_A = self.teacher_model.netG_B(self.real_B)  # G_B(B)

            self.teacher_model.netD_A(self.fake_B)
            self.teacher_model.netD_B(self.fake_A)

            if self.opt.lambda_discriminator_distill > 0:
                self.teacher_model_discriminator.netD_A(teacher_fake_B)
                self.teacher_model_discriminator.netD_B(teacher_fake_A)

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
        # attention distill loss
        self.loss_attention_distill = 0.0
        if self.opt.lambda_attention_distill > 0:
            self.loss_attention_distill = self.distill_attention_loss() * self.opt.lambda_attention_distill
        # discriminator distill loss
        self.loss_discriminator_distill = 0.0
        if self.opt.lambda_discriminator_distill > 0:
            self.loss_discriminator_distill = self.distill_discriminator_loss() * self.opt.lambda_discriminator_distill
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + \
                      self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + \
                      self.loss_attention_distill + self.loss_discriminator_distill
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
            'G_A': self.__pop_ops_params_state_dict(self.netG_A.state_dict()),
            'G_B': self.__pop_ops_params_state_dict(self.netG_B.state_dict()),
            'D_A': self.netD_A.state_dict(),
            'D_B': self.netD_B.state_dict(),
            'epoch': epoch,
            'cfg': (self.cfg_AtoB, self.cfg_BtoA),
            'fid': fid
        }
        if isbest:
            torch.save(ckpt, os.path.join(save_dir, 'model_best_%s.pth' % direction))
        else:
            torch.save(ckpt, os.path.join(save_dir, 'model_%d.pth' % epoch))

    def __pop_ops_params_state_dict(self, state_dict):

        for k in list(state_dict.keys()):
            if str.endswith(k, 'total_ops') or str.endswith(k, 'total_params'):
                state_dict.pop(k)
        return state_dict

    def load_models(self, load_path):
        ckpt = torch.load(load_path, map_location=self.device)
        self.netG_A.load_state_dict(self.__pop_ops_params_state_dict(ckpt['G_A']))
        self.netG_B.load_state_dict(self.__pop_ops_params_state_dict(ckpt['G_B']))
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

    def init_attention_distill(self):
        if self.opt.pretrain_path is None or not os.path.exists(self.opt.pretrain_path):
            raise FileExistsError('The pretrain model path must be exist!!!')
        new_opt = copy.copy(self.opt)
        new_opt.ngf = 64
        new_opt.lambda_attention_distill = 0.0
        new_opt.lambda_discriminator_distill = 0.0
        self.teacher_model = MobileCycleGANModel(new_opt)
        self.teacher_model.load_models(self.opt.pretrain_path)

        self.loss_names.append('attention_distill')

        self.total_feature_out_AtoB_teacher = {}
        self.total_feature_out_BtoA_teacher = {}
        self.total_feature_out_AtoB_student = {}
        self.total_feature_out_BtoA_student = {}
        self.total_feature_out_DA_teacher = {}
        self.total_feature_out_DB_teacher = {}

        self.teacher_extract_G_layers = ['model.9', 'model.12', 'model.15', 'model.18']
        self.teacher_extract_D_layers = ['model.4']#, 'model.10']
        self.student_extract_G_layers = ['model.9', 'model.12', 'model.15', 'model.18']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output
            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self.teacher_model.netG_A, self.total_feature_out_AtoB_teacher, self.teacher_extract_G_layers)
        add_hook(self.teacher_model.netG_B, self.total_feature_out_BtoA_teacher, self.teacher_extract_G_layers)
        add_hook(self.teacher_model.netD_A, self.total_feature_out_DA_teacher, self.teacher_extract_D_layers)
        add_hook(self.teacher_model.netD_B, self.total_feature_out_DB_teacher, self.teacher_extract_D_layers)
        add_hook(self.netG_A, self.total_feature_out_AtoB_student, self.student_extract_G_layers)
        add_hook(self.netG_B, self.total_feature_out_BtoA_student, self.student_extract_G_layers)

    def distill_attention_loss(self):

        total_attention_AtoB_teacher = [f.pow(2).mean(1, keepdim=True) for f in
                                        self.total_feature_out_AtoB_teacher.values()]
        total_attention_BtoA_teacher = [f.pow(2).mean(1, keepdim=True) for f in
                                        self.total_feature_out_BtoA_teacher.values()]
        total_attention_DA_teacher = [f.pow(2).mean(1, keepdim=True) for f in
                                      self.total_feature_out_DA_teacher.values()]
        total_attention_DB_teacher = [f.pow(2).mean(1, keepdim=True) for f in
                                      self.total_feature_out_DB_teacher.values()]
        total_attention_AtoB_student = [f.pow(2).mean(1, keepdim=True) for f in
                                        self.total_feature_out_AtoB_student.values()]
        total_attention_BtoA_student = [f.pow(2).mean(1, keepdim=True) for f in
                                        self.total_feature_out_BtoA_student.values()]

        # total_attention_DA_teacher[1] = util.attention_interpolate(
        #     total_attention_DA_teacher[1])  # interpolate attention map from 31*31 to 64*64
        # total_attention_DB_teacher[1] = util.attention_interpolate(total_attention_DB_teacher[1])

        total_mixup_attention_AtoB = []
        total_mixup_attention_BtoA = []

        for i in range(len(total_attention_AtoB_teacher)):
            total_mixup_attention_AtoB.append(util.mixup_attention(
                [total_attention_AtoB_teacher[i], total_attention_DA_teacher[0]],
                [0.5, 0.5]))
            total_mixup_attention_BtoA.append(util.mixup_attention(
                [total_attention_BtoA_teacher[i], total_attention_DB_teacher[0]],
                [0.5, 0.5]))

        total_distill_loss = 0.0
        for i in range(len(total_attention_AtoB_teacher)):
            total_distill_loss += util.attention_loss(total_mixup_attention_AtoB[i], total_attention_AtoB_student[i],
                                                      normalize=self.opt.attention_normal)
            total_distill_loss += util.attention_loss(total_mixup_attention_BtoA[i], total_attention_BtoA_student[i],
                                                      normalize=self.opt.attention_normal)
        return total_distill_loss

    def init_discriminator_distill(self):
        if self.opt.pretrain_path is None or not os.path.exists(self.opt.pretrain_path):
            raise FileExistsError('The pretrain model path must be exist!!!')
        new_opt = copy.copy(self.opt)
        new_opt.ngf = 64
        new_opt.lambda_attention_distill = 0.0
        new_opt.lambda_discriminator_distill = 0.0
        if self.teacher_model is None:
            self.teacher_model = MobileCycleGANModel(new_opt)
            self.teacher_model.load_models(self.opt.pretrain_path)
        self.teacher_model_discriminator = MobileCycleGANModel(new_opt)
        self.teacher_model_discriminator.load_models(self.opt.pretrain_path)

        self.loss_names.append('discriminator_distill')

        self.total_feature_out_DA_teacher_discriminator = {}
        self.total_feature_out_DB_teacher_discriminator = {}
        self.total_feature_out_DA_student_discriminator = {}
        self.total_feature_out_DB_student_discriminator = {}

        self.teacher_extract_D_layers_discriminator = ['model.10']

        def get_activation(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output
            return get_output_hook

        def add_hook(model, maps, extract_layers):
            for name, module in model.named_modules():
                if name in extract_layers:
                    module.register_forward_hook(get_activation(maps, name))

        add_hook(self.teacher_model.netD_A, self.total_feature_out_DA_teacher_discriminator,
                 self.teacher_extract_D_layers_discriminator)
        add_hook(self.teacher_model.netD_B, self.total_feature_out_DB_teacher_discriminator,
                 self.teacher_extract_D_layers_discriminator)
        add_hook(self.teacher_model_discriminator.netD_A, self.total_feature_out_DA_student_discriminator,
                 self.teacher_extract_D_layers_discriminator)
        add_hook(self.teacher_model_discriminator.netD_B, self.total_feature_out_DB_student_discriminator,
                 self.teacher_extract_D_layers_discriminator)

    def distill_discriminator_loss(self):

        total_distill_loss = 0.0
        for i in self.teacher_extract_D_layers_discriminator:
            total_distill_loss += torch.norm(self.total_feature_out_DA_teacher_discriminator[i] - self.total_feature_out_DA_student_discriminator[i], 2)
            total_distill_loss += torch.norm(self.total_feature_out_DB_teacher_discriminator[i] - self.total_feature_out_DB_student_discriminator[i], 2)
        return total_distill_loss


# class opts():
#     mask_loss_type = 'relu'
#     upconv_solo = True,
#     upconv_bound = True,
#     upconv_relu = False,
#     gpu_ids = []
# cfg = None
# # cfg = [64, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 128, 64]
# moblie_model = MobileResnetGenerator(input_nc=3, output_nc=3, ngf=64, opt=opts(), cfg=cfg)
# print(moblie_model)
#
# for k, v in moblie_model.state_dict().items():
#     print(k, v.size())
#
# input = torch.randn((1, 3, 256, 256))
# print(moblie_model(input).size())
# from thop import profile
# macs, params = profile(moblie_model, (input, ))
# print(macs, params)