import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from scipy.interpolate import lagrange

class Mask(nn.Module):

    def __init__(self, out_channels, mask_loss_type='relu'):
        super(Mask, self).__init__()

        self.mask_loss_type = mask_loss_type

        self.bound = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.mask_weight = nn.Parameter(torch.Tensor(out_channels), requires_grad=True)
        self.reset_parameter()
        self.update(bound=1.0)

    def reset_parameter(self):
        init.normal_(self.mask_weight.data, 0.0, 0.2)

    def get_block_decay_loss(self, block_sparsity_coeff):
        if self.bound == 0.5:
            return 0

        relu_mask = (self.mask_weight > -self.bound) & (self.mask_weight < self.bound)
        return torch.sum(relu_mask.float() * (self.mask_weight + self.bound) * block_sparsity_coeff)
    def get_weight_decay_loss(self):

        if self.bound == 0.0:
            return 0.0

        if self.mask_loss_type == 'bound':
            bound_mask = (self.mask_weight > -self.bound) & (self.mask_weight < self.bound)
            return torch.norm(self.bound - torch.abs(self.mask_weight[bound_mask]), p=2)
        elif self.mask_loss_type == 'relu':
            relu_mask = (self.mask_weight > -self.bound) & (self.mask_weight < self.bound)
            return torch.sum(relu_mask.float() * (self.mask_weight + self.bound))
        elif self.mask_loss_type == 'uprelu':
            relu_mask = (self.mask_weight > -self.bound) & (self.mask_weight < self.bound)
            return torch.sum(relu_mask.float() * (self.bound - self.mask_weight))
        else:
            return 0

    def update(self, bound):

        self.bound.data.fill_(bound)

        if bound > 0.0 and bound < 0.005:
            bound = 0.005
        xs = [-bound, -bound / 2, 0.0]
        ys = [0.0, 0.125, 0.5]
        left_lagrange_params = list(lagrange(xs, ys).c)

        xs = [0.0, bound / 2, bound]
        ys = [0.5, 0.875, 1.0]
        right_lagrange_params = list(lagrange(xs, ys).c)

        left_lagrange_params.extend(right_lagrange_params)
        self.stepfunc_params = torch.FloatTensor(left_lagrange_params).to(self.bound.device)

    def get_current_mask(self):

        if self.bound != 0.0:
            mask = torch.zeros(self.mask_weight.size()).to(self.mask_weight.device)
            one_mask = self.mask_weight > self.bound
            left_mask = (self.mask_weight >= -self.bound) & (self.mask_weight <= 0.0)
            right_mask = (self.mask_weight > 0.0) & (self.mask_weight <= self.bound)
            mask[one_mask] = 1.0
            mask[left_mask] = (self.mask_weight[left_mask] * self.stepfunc_params[0] + self.stepfunc_params[1]) * \
                              self.mask_weight[left_mask] + self.stepfunc_params[2]
            mask[right_mask] = (self.mask_weight[right_mask] * self.stepfunc_params[3] + self.stepfunc_params[4]) * \
                               self.mask_weight[right_mask] + self.stepfunc_params[5]
        else:
            mask = (self.mask_weight > self.bound).float()
        return mask

    def forward(self, x):
        output = Step_function.apply(x, self.mask_weight, self.bound, self.stepfunc_params)
        return output

class Step_function(torch.autograd.Function):

    @staticmethod
    def forward(self, input, mask_weight, bound, stepfunc_params):

        if bound != 0.0:
            mask = torch.zeros(mask_weight.size()).to(mask_weight.device)
            one_mask = mask_weight > bound
            left_mask = (mask_weight >= -bound) & (mask_weight <= 0.0)
            right_mask = (mask_weight > 0.0) & (mask_weight <= bound)
            mask[one_mask] = 1.0
            mask[left_mask] = (mask_weight[left_mask] * stepfunc_params[0] + stepfunc_params[1]) *\
                                    mask_weight[left_mask] + stepfunc_params[2]
            mask[right_mask] = (mask_weight[right_mask] * stepfunc_params[3] + stepfunc_params[4]) * \
                                   mask_weight[right_mask] + stepfunc_params[5]
        else:
            mask = (mask_weight > bound).float()

        self.save_for_backward(input, mask_weight, mask, bound, stepfunc_params)
        return input * mask[None, :, None, None]

    @staticmethod
    def backward(self, grad_output):
        input, mask_weight, mask, bound, stepfunc_params = self.saved_tensors

        d_mask = torch.zeros(mask_weight.size()).to(mask_weight.device)
        if bound != 0.0:
            left_mask = (mask_weight >= -bound) & (mask_weight <= 0.0)
            right_mask = (mask_weight > 0.0) & (mask_weight <= bound)
            d_mask[left_mask] = mask_weight[left_mask] * 2 * stepfunc_params[0] + stepfunc_params[1]
            d_mask[right_mask] = mask_weight[right_mask] * 2 * stepfunc_params[3] + stepfunc_params[4]

        grad_mask_weight = torch.sum(grad_output * input * d_mask[None, :, None, None], dim=(0, 2, 3))
        grad_input = grad_output * mask[None, :, None, None]

        return Variable(grad_input), Variable(grad_mask_weight), None, None