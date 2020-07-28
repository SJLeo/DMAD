"""This module contains simple helper functions """
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import lr_scheduler

import numpy as np
from PIL import Image
import os
import cv2

import logging
import shutil

def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled

def tensor2imgs(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2imgs(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2imgs(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def attention_interpolate(input_image, size=(64, 64), interp_method=cv2.INTER_AREA):

    image = tensor2im(input_image)
    # save_image(image, './image.png')
    interp_image = cv2.resize(image, size, interpolation=interp_method)
    interp_image = interp_image[:, :, np.newaxis]
    interp_image = np.transpose((interp_image / 255.0 * 2.0 - 1), (2, 0, 1))
    interp_image = interp_image[np.newaxis, :, :, :]
    return torch.FloatTensor(interp_image).to(input_image.device)

def attention_loss(attention_x, attention_y, normalize=True):

    if normalize:
        return (F.normalize(attention_x.view(attention_x.size(0), -1)) -
                F.normalize(attention_y.view(attention_y.size(0), -1))).pow(2).mean()
        # return (attention_x/torch.norm(attention_x, 2) - attention_y/torch.norm(attention_y, 2)).pow(2).mean()
    else:
        return (attention_x - attention_y).pow(2).mean()

def mixup_attention(attention_maps, lams):

    if len(attention_maps) != len(lams):
        raise IndexError('Mixup attention map dim must equal to lams')

    x = attention_maps[0] * lams[0]
    for i in range(1, len(attention_maps), 1):
        x += attention_maps[i] * lams[i]

    return x

def group_lasso_loss(group_weights, group=256):

    loss = 0.
    for i in range(group):
        loss += torch.norm(group_weights[:, i], 2)
    return loss

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def load_network(net, load_path, verbose=True):
    if verbose:
        print('Load network at %s' % load_path)
    weights = torch.load(load_path)
    if isinstance(net, nn.DataParallel):
        net = net.module
    net.load_state_dict(weights)
    return net


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def save_images(visuals, img_path, save_image_dir, direction='AtoB', aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    imageA_path = img_path[0][0] if direction == 'AtoB' else img_path[1][0]
    imageB_path = img_path[1][0] if direction == 'AtoB' else img_path[0][0]
    imageA_name = (imageA_path.split('/')[-1].split('\\')[-1]).split('.')[0]
    imageB_name = (imageB_path.split('/')[-1].split('\\')[-1]).split('.')[0]

    for label, im_data in visuals.items():
        if label == 'fake_B' or label == 'fake_A':
            im = tensor2im(im_data)

            image_name = '%s_%s.png' % (imageB_name if label=='fake_A' else imageA_name, label)
            save_path = os.path.join(save_image_dir, label)
            mkdirs(save_path)
            save_image(im, os.path.join(save_path, image_name), aspect_ratio=aspect_ratio)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_logger(file_path):
    logger = logging.getLogger('Mask-GAN')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.normal_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def combine_best_model(best_AtoB_epoch, best_BtoA_epoch, source_path, target_path, type):

    if type == 'cyclegan' or type == 'mobilecyclegan':
        best_AtoB_model = torch.load(os.path.join(source_path, 'model_%d.pth' % best_AtoB_epoch), map_location='cpu')
        best_BtoA_model = torch.load(os.path.join(source_path, 'model_%d.pth' % best_BtoA_epoch), map_location='cpu')

        best_ckpt = {
            'G_A': best_AtoB_model['G_A'],
            'G_B': best_BtoA_model['G_B'],
            'D_A': best_AtoB_model['D_A'],
            'D_B': best_BtoA_model['D_B'],
            'cfg': (best_AtoB_model['cfg'][0], best_BtoA_model['cfg'][1]),
            'fid': (best_AtoB_model['fid'][0], best_BtoA_model['fid'][1])
        }
        torch.save(best_ckpt, os.path.join(target_path, 'model_best.pth'))
    elif type == 'pix2pix' or type == 'mobilepix2pix':
        best_AtoB_model = torch.load(os.path.join(source_path, 'model_%d.pth' % best_AtoB_epoch), map_location='cpu')
        best_ckpt = {
            'G': best_AtoB_model['G'],
            'D': best_AtoB_model['D'],
            'cfg': best_AtoB_model['cfg'],
            'fid': best_AtoB_model['fid']
        }
        torch.save(best_ckpt, os.path.join(target_path, 'model_best.pth'))
    shutil.rmtree(source_path)