# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import argparse
import os
import pprint

import logging
import timeit
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
from models import module_dict
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BoundaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel, SimpleModel, calculate_FPS


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)        
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.ENV.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    model = models.BEVANet.get_model(config, task='seg')
    
    input_size = config['TEST']['IMAGE_SIZE']
    logger.info('FPS: ' + str(calculate_FPS(input_tensor=torch.randn(1, 3, input_size[0], input_size[1]), model=model)))
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.DETAIL.MULTI_SCALE,
                        smooth_kernel=config.TRAIN.DETAIL.SMOOTH_KERNEL,
                        to_binary=config.TRAIN.DETAIL.TO_BINARY,
                        rand_scale=config.TRAIN.RAND_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.ENV.WORKERS,
        pin_memory=False,
        drop_last=True)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        smooth_kernel=0,
                        to_binary=True,
                        rand_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.ENV.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

    boundary_criterion = BoundaryLoss(loss_type=config.LOSS.BCE_TYPE)
    
    if config.MODEL.BRANCHES > 1:
        model = FullModel(model, sem_criterion, boundary_criterion)
    else:
        model = SimpleModel(model, sem_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
    # msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}'.format(
    #                     valid_loss, mean_IoU)
    # print(msg)
    # exit(0)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    elif config.TRAIN.OPTIMIZER == 'adamw':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params,
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WD
        )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    CHECKPOINT_PATH = os.path.join(final_output_dir, 'checkpoint.pth.tar')
    if config.TRAIN.RESUME:
        model_state_file = CHECKPOINT_PATH
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    valid_epoch_freq = config.VALID.VALID_EPOCH_FREQ
    valid_last_epoch = config.VALID.VALID_LAST_EPOCH
    real_end = end_epoch
    valid_last_start_epoch = real_end - valid_last_epoch
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        if flag_rm == 1 or (epoch % valid_epoch_freq == 0 and epoch < valid_last_start_epoch) or (epoch >= valid_last_start_epoch):
            valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)

            logger.info('=> saving checkpoint to {}'.format(CHECKPOINT_PATH))
            torch.save({
                    'epoch': epoch+1,
                    'best_mIoU': best_mIoU,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, CHECKPOINT_PATH
            )
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best.pt'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

        if flag_rm == 1:
            flag_rm = 0

    torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Time: %s' % str(datetime.timedelta(seconds=end-start)))
    logger.info('Done')

if __name__ == '__main__':
    main()
