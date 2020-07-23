import torch
import torch.nn as nn

from models.GANLoss import GANLoss
import utils.util as util

import functools
import os
from collections import OrderedDict

class UnetSkipConnectionBlock(nn.Module):

    # def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
    #              outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
    #     super(UnetSkipConnectionBlock, self).__init__()
    #     self.outermost = outermost
    #     norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    #     if type(norm_layer) == functools.partial:
    #         use_bias = norm_layer.func == nn.InstanceNorm2d
    #     else:
    #         use_bias = norm_layer == nn.InstanceNorm2d
    #     if input_nc is None:
    #         input_nc = outer_nc
    #     downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
    #                          stride=2, padding=1, bias=use_bias)
    #     downrelu = nn.LeakyReLU(0.2, True)
    #     downnorm = norm_layer(inner_nc)
    #     uprelu = nn.ReLU(True)
    #     upnorm = norm_layer(outer_nc)
    #
    #     if outermost:
    #         upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
    #                                     kernel_size=4, stride=2,
    #                                     padding=1)
    #         down = [downconv]
    #         up = [uprelu, upconv, nn.Tanh()]
    #         model = down + [submodule] + up
    #     elif innermost:
    #         upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
    #                                     kernel_size=4, stride=2,
    #                                     padding=1, bias=use_bias)
    #         down = [downrelu, downconv]
    #         up = [uprelu, upconv, upnorm]
    #         model = down + up
    #     else:
    #         upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
    #                                     kernel_size=4, stride=2,
    #                                     padding=1, bias=use_bias)
    #         down = [downrelu, downconv, downnorm]
    #         up = [uprelu, upconv, upnorm]
    #
    #         if use_dropout:
    #             model = down + [submodule] + up + [nn.Dropout(0.5)]
    #         else:
    #             model = down + [submodule] + up
    #
    #     self.model = nn.Sequential(*model)

    def __init__(self, conv_inchannel, conv_outchannel, upconv_inchannel, upconv_outchannel, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        downconv = nn.Conv2d(conv_inchannel, conv_outchannel, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(conv_outchannel)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(upconv_outchannel)

        if outermost:
            upconv = nn.ConvTranspose2d(upconv_inchannel, upconv_outchannel,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(upconv_inchannel, upconv_outchannel,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(upconv_inchannel, upconv_outchannel,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

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

class UnetGenertor(nn.Module):

    # def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, filter_cfg=None, channel_cfg=None):
    #     super(UnetGenertor, self).__init__()
    #
    #
    #     unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
    #                                          norm_layer=norm_layer, innermost=True)
    #     for i in range(num_downs - 5):
    #         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
    #                                              norm_layer=norm_layer, use_dropout=use_dropout)
    #
    #     unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
    #                                          norm_layer=norm_layer, use_dropout=use_dropout)
    #     unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
    #                                          norm_layer=norm_layer, use_dropout=use_dropout)
    #     unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
    #                                          norm_layer=norm_layer, use_dropout=use_dropout)
    #     self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
    #                                          norm_layer=norm_layer)

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 filter_cfgs=None, channel_cfgs=None):
        super(UnetGenertor, self).__init__()

        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[6],
                                             conv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[6],
                                             upconv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[7],
                                             upconv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[7],
                                             submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[5-i],
                                             conv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[5-i],
                                             upconv_inchannel=ngf * 16 if channel_cfgs is None else channel_cfgs[8+i],
                                             upconv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[8+i],
                                             submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 4 if channel_cfgs is None else channel_cfgs[2],
                                             conv_outchannel=ngf * 8 if filter_cfgs is None else filter_cfgs[2],
                                             upconv_inchannel=ngf * 16 if channel_cfgs is None else channel_cfgs[11],
                                             upconv_outchannel=ngf * 4 if filter_cfgs is None else filter_cfgs[11],
                                             submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf * 2 if channel_cfgs is None else channel_cfgs[1],
                                             conv_outchannel=ngf * 4 if filter_cfgs is None else filter_cfgs[1],
                                             upconv_inchannel=ngf * 8 if channel_cfgs is None else channel_cfgs[12],
                                             upconv_outchannel=ngf * 2 if filter_cfgs is None else filter_cfgs[12],
                                             submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock(conv_inchannel=ngf if channel_cfgs is None else channel_cfgs[0],
                                             conv_outchannel=ngf * 2 if filter_cfgs is None else filter_cfgs[0],
                                             upconv_inchannel=ngf * 4 if channel_cfgs is None else channel_cfgs[13],
                                             upconv_outchannel=ngf if filter_cfgs is None else filter_cfgs[13],
                                             submodule=unet_block, norm_layer=norm_layer)

        self.model = UnetSkipConnectionBlock(conv_inchannel=input_nc,
                                conv_outchannel=ngf,
                                upconv_inchannel=ngf * 2 if channel_cfgs is None else channel_cfgs[14],
                                upconv_outchannel=output_nc,
                                submodule=unet_block, norm_layer=norm_layer, outermost=True)

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

class Pix2PixModel(nn.Module):

    def __init__(self, opt, filter_cfgs=None, channel_cfgs=None):
        super(Pix2PixModel, self).__init__()

        self.opt = opt
        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if len(opt.gpu_ids) > 0 else 'cpu'
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.netG = UnetGenertor(input_nc=3, output_nc=3, num_downs=8, ngf=opt.ngf, use_dropout=not opt.no_dropout,
                                 filter_cfgs=filter_cfgs, channel_cfgs=channel_cfgs)
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
        # print(ckpt.keys())
        self.netG.load_state_dict(ckpt['G'])
        self.netD.load_state_dict(ckpt['D'])

        # print('loading the model from %s' % load_path)
        print('loading the epoch %d model from %s' % (ckpt['epoch'], load_path))
        return ckpt['fid'], float('inf')
        # return 0, 0

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

    def register_hook(self):

        self.total_feature_out_G = {}
        self.total_feature_out_D = {}

        def get_activate(maps, name):
            def get_output_hook(module, input, output):
                maps[name] = output
            return get_output_hook

        for name, module in self.netG.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.register_forward_hook(get_activate(self.total_feature_out_G, name))