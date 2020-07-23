import torch
import torch.nn as nn

class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('zero_tensor', torch.tensor(0.))
        self.zero_tensor.requires_grad_(False)
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = None
        elif gan_mode == ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def get_zero_tensor(self, prediction):
        return self.zero_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(prediction - 1, self.get_zero_tensor(prediction))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-prediction - 1, self.get_zero_tensor(prediction))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real
                loss = -torch.mean(prediction)
        return loss