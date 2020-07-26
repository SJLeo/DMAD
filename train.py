import torch

from options import options
from data import create_dataset
from models import CycleGAN, MaskCycleGAN, Pix2Pix, MaskPix2Pix
from models import MobileCycleGAN, MaskMobileCycleGAN, MobilePix2Pix, MaskMobilePix2Pix
from utils.visualizer import Visualizer
import utils.util as util
from metric import get_fid, get_mIoU
from metric.inception import InceptionV3
from metric.mIoU_score import DRNSeg

import time
import ntpath
import os
import copy
import numpy as np

def test_cyclegan_fid(model, opt):
    opt.phase = 'test'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    opt.display_id = -1
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_A = {}
    fake_B = {}

    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_B[data['A_paths'][0]] = visuals['fake_B']
        fake_A[data['B_paths'][0]] = visuals['fake_A']
        util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)

    # print('Calculating AtoB FID...', flush=True)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat_B.npz'))
    AtoB_fid = get_fid(list(fake_B.values()), inception_model, npz, model.device, opt.batch_size)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat_A.npz'))
    BtoA_fid = get_fid(list(fake_A.values()), inception_model, npz, model.device, opt.batch_size)

    return AtoB_fid, BtoA_fid

def test_pix2pix_fid(model, opt):
    opt.phase = 'val'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    opt.display_id = -1
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_B = {}
    for i, data in enumerate(dataset):
        model.set_input(data)
        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_B[data['A_paths'][0]] = visuals['fake_B']
        util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_model = InceptionV3([block_idx])
    inception_model.to(model.device)
    inception_model.eval()
    npz = np.load(os.path.join(opt.dataroot, 'real_stat_B.npz'))
    fid = get_fid(list(fake_B.values()), inception_model, npz, model.device, opt.batch_size)

    return fid

def test_pix2pix_mIoU(model, opt):
    opt.phase = 'val'
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.load_size = 256
    opt.display_id = -1
    dataset = create_dataset(opt)
    model.model_eval()

    result_dir = os.path.join(opt.checkpoints_dir, opt.name, 'test_results')
    util.mkdirs(result_dir)

    fake_B = {}
    names = set()
    for i, data in enumerate(dataset):
        model.set_input(data)

        with torch.no_grad():
            model.forward()
        visuals = model.get_current_visuals()
        fake_B[data['A_paths'][0]] = visuals['fake_B']

        for path in range(len(model.image_paths)):
            short_path = ntpath.basename(model.image_paths[0][0])
            name = os.path.splitext(short_path)[0]
            names.add(name)
        util.save_images(visuals, model.image_paths, result_dir, direction=opt.direction,
                         aspect_ratio=opt.aspect_ratio)

    drn_model = DRNSeg('drn_d_105', 19, pretrained=False).to(model.device)
    util.load_network(drn_model, opt.drn_path, verbose=False)
    drn_model.eval()

    mIoU = get_mIoU(list(fake_B.values()), names, drn_model, model.device,
                    table_path=os.path.join(opt.dataroot, 'table.txt'),
                    data_dir=opt.dataroot,
                    batch_size=opt.batch_size,
                    num_workers=opt.num_threads)
    return mIoU

if __name__ == '__main__':

    best_AtoB_fid = float('inf')
    best_BtoA_fid = float('inf')
    best_AtoB_epoch = 0
    best_BtoA_epoch = 0

    opt = options.parse()
    opt.isTrain = True
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
    logger = util.get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'logger.log'))

    # create model
    if opt.model == 'cyclegan':
        if opt.mask:
            model = MaskCycleGAN.MaskCycleGANModel(opt)
        else:
            model = CycleGAN.CycleGANModel(opt)
    elif opt.model == 'pix2pix':
        opt.norm = 'batch'
        opt.dataset_mode = 'aligned'
        opt.pool_size = 0
        if opt.mask:
            model = MaskPix2Pix.MaskPix2PixModel(opt)
        else:
            model = Pix2Pix.Pix2PixModel(opt)
    elif opt.model == 'mobilecyclegan':
        if opt.mask:
            model = MaskMobileCycleGAN.MaskMobileCycleGANModel(opt)
        else:
            model = MobileCycleGAN.MobileCycleGANModel(opt)
    elif opt.model == 'mobilepix2pix':
        opt.norm = 'batch'
        opt.dataset_mode = 'aligned'
        opt.pool_size = 0
        if opt.mask:
            model = MaskMobilePix2Pix.MaskMobilePix2PixModel(opt)
        else:
            model = MobilePix2Pix.MobilePix2PixModel(opt)
    else:
        raise NotImplementedError('%s not implemented' % opt.model)

    if opt.continue_train:
        if opt.load_path is None or not os.path.exists(opt.load_path):
            raise FileExistsError('Load path must be exist!!!')
        best_fid_AtoB, best_fid_BtoA = model.load_models(opt.load_path)

    # create dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    logger.info('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)
    total_iters = 0
    all_total_iters = dataset_size * opt.batch_size * (opt.n_epochs + opt.n_epochs_decay)
    update_bound_freq = all_total_iters * 0.75 // 150

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

        model.model_train()
        logger.info('\nEpoch:%d' % epoch)

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        for i, data in enumerate(dataset):

            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                # model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % update_bound_freq == 0 and opt.mask:
                model.update_masklayer(current_iter=total_iters, all_total_iters=all_total_iters)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                loss_message = visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                logger.info(loss_message)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / (dataset_size * opt.batch_size), losses)

                iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:

            if opt.model == 'cyclegan' or opt.model == 'mobilecyclegan':
                AtoB_fid, BtoA_fid = test_cyclegan_fid(model, copy.copy(opt))
                fid = (AtoB_fid, BtoA_fid)
                logger.info('AtoB FID: %.2f' % AtoB_fid)
                logger.info('BtoA FID: %.2f' % BtoA_fid)

                if (opt.mask and epoch > (opt.n_epochs + opt.n_epochs_decay) * 0.75) or not opt.mask:
                    if AtoB_fid < best_AtoB_fid:
                        model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                          fid=fid, isbest=True, direction='AtoB')
                        best_AtoB_fid = AtoB_fid
                        best_AtoB_epoch = epoch
                    if BtoA_fid < best_BtoA_fid:
                        model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                          fid=fid, isbest=True, direction='BtoA')
                        best_BtoA_fid = BtoA_fid
                        best_BtoA_epoch = epoch

            elif opt.model == 'pix2pix' or opt.model == 'mobilepix2pix':

                if 'cityscapes' in opt.dataroot:
                    mIoU = test_pix2pix_mIoU(model, copy.copy(opt))
                    logger.info('mAP: %.2f' % mIoU)
                    fid = mIoU

                    if mIoU > best_BtoA_fid and (not opt.mask or (opt.mask and epoch > (opt.n_epochs + opt.n_epochs_decay) * 0.75)):
                        model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                          fid=fid, isbest=True, direction=opt.direction)
                        best_BtoA_fid = fid
                        best_BtoA_epoch = epoch
                else:
                    fid = test_pix2pix_fid(model, copy.copy(opt))
                    logger.info('FID: %.2f' % fid)

                    if fid < best_AtoB_fid and (not opt.mask or (opt.mask and epoch > (opt.n_epochs + opt.n_epochs_decay) * 0.75)):
                        model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                                          fid=fid, isbest=True, direction='AtoB')
                        best_AtoB_fid = fid
                        best_AtoB_epoch = epoch

            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_models(epoch, os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'), fid=fid)

        logger.info('End of epoch %d / %d \t Time Taken: %d sec' %  (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

        if opt.mask:
            model.print_sparsity_info(logger)

        model.update_learning_rate(epoch)  # update learning rates at the end of every epoch.

    logger.info('Best AtoB Epoch %d:%.2f' % (best_AtoB_epoch, best_AtoB_fid))
    logger.info('Best BtoA Epoch %d:%.2f' % (best_BtoA_epoch, best_BtoA_fid))

    util.combine_best_model(best_AtoB_epoch=best_AtoB_epoch,
                            best_BtoA_epoch=best_BtoA_epoch,
                            source_path=os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints'),
                            target_path=os.path.join(opt.checkpoints_dir, opt.name),
                            type=opt.model)
    logger.info('Best model save in %s' % os.path.join(opt.checkpoints_dir, opt.name, 'model_best.pth'))