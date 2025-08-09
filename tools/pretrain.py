# ------------------------------------------------------------------------------
# Modified based on https://github.com/pytorch/examples/blob/main/imagenet/main.py
# ------------------------------------------------------------------------------
import argparse
import os
import re
import random
import shutil
import time
import warnings
from enum import Enum
import ast
import logging
import pprint
import timeit
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from torch.utils.data import Subset
from torchvision.transforms import v2
from torch.utils.data import default_collate

import _init_paths
from datasets.imagenet import ImageNet
from configs import config
from configs import update_config
import models
from utils.utils import calculate_FPS


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--cfg', help='experiment configure file name',
                    default="configs/cityscapes/BEVANet.yaml", type=str)
parser.add_argument('opts', help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
parser.add_argument('data_root', metavar='DIR', nargs='?', default='data/imagenet',
                    help='path to dataset (default: data/imagenet)')
parser.add_argument('ckpt_path', metavar='DIR', nargs='?', default='pretrained_models/imagenet',
                    help='path to dataset (default: pretrained_models/imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=135, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='optimizer', help='the optimizer', dest='optimizer')
parser.add_argument('--scheduler', default='step', type=str,
                    metavar='scheduler', help='the scheduler', dest='scheduler')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 1000)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=88, type=int,
                    help='seed for initializing training. (default: 88)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc = 0
time_str = time.strftime('%Y-%m-%d-%H-%M')

def main():
    args = parser.parse_args()
    update_config(config, args)

    use_cuda = torch.cuda.is_available()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed)     
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if use_cuda:
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    logger = create_logger()
    
    os.makedirs(os.path.join(args.ckpt_path, config.MODEL.NAME, time_str), exist_ok=True)
    logger.info(pprint.pformat(args))
    logger.info(config)
    logger.info("model: " + config.MODEL.NAME)
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    
    model = models.BEVANet.get_model(config, task='cls')
    logger.info('FPS: ' + str(calculate_FPS(input_tensor=torch.randn(1, 3, 640, 640), model=model)))

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Data loading code
    train_dataset = ImageNet(args.data_root, split='train').get_dataset()
    val_dataset = ImageNet(args.data_root, split='val').get_dataset()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    cutmix = v2.CutMix(num_classes=1000, alpha=args.cutmix)
    mixup = v2.MixUp(num_classes=1000, alpha=args.mixup)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        mixed_images, mixed_labels = cutmix_or_mixup(images, labels)
        return mixed_images, mixed_labels, labels
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.scheduler == "cosine":
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: warmup_lambda(epoch, args.warmup_epochs, args.lr, args.warmup_lr))
        total_iterations = (args.epochs - args.warmup_epochs) * len(train_loader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations, eta_min=args.min_lr)
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                if isinstance(best_acc, float):
                    best_acc = torch.tensor(best_acc)
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if args.evaluate:
        logger.info('\nEvaluation only')
        acc = validate(val_loader, model, criterion, logger, args)
        logger.info(' Valid Acc:  %.2f' % (acc))
        return

    start = timeit.default_timer()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, logger, args)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, logger, args)
        scheduler.step()

        if args.optimizer == 'cosine' and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args=args)

    end = timeit.default_timer()
    logger.info('Time: %s' % str(datetime.timedelta(seconds=end-start)))
    logger.info('Done')


def warmup_lambda(epoch, warmup_epochs, initial_lr, warmup_lr):
    if epoch < warmup_epochs:
        return (warmup_lr + (initial_lr - warmup_lr) * (epoch / warmup_epochs)) / initial_lr
    return 1.0

def preprocess_string(s):
    # Use regular expression to find all identifiers (variable names)
    # and replace them with quoted strings
    def replacer(match):
        identifier = match.group(0)
        # Check if the identifier is a known literal (like None)
        if identifier in {'None', 'True', 'False'}:
            return identifier
        # Otherwise, replace with quoted string
        return f'"{identifier}"'

    # Match identifiers that are not within quotes
    processed_string = re.sub(r'\b\w+\b', replacer, s)
    return processed_string


def cast_string(value):
    if value is None or value == "None":
        return None
    try:
        # Try to parse the string as a list using ast.literal_eval
        result = ast.literal_eval(preprocess_string(value))
        if isinstance(result, list) and all(isinstance(item, str) or item is None for item in result):
            return result
    except (ValueError, SyntaxError):
        pass
    # If it's not None and not a list, return the original string
    return value


def train(train_loader, model, criterion, optimizer, epoch, device, logger, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    eta = int(((args.epochs - epoch - 1) * len(train_loader) * batch_time.avg))
    eta = str(datetime.timedelta(seconds=eta))
    logger.info("ETA: " + eta)


def validate(val_loader, model, criterion, logger, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if i % args.print_freq == 0:
                #     progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ',
        logger=logger)

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth'):
    torch.save(state, os.path.join(args.ckpt_path, config.MODEL.NAME, time_str, filename))
    if is_best:
        torch.save(state, os.path.join(args.ckpt_path, config.MODEL.NAME, time_str, 'model_best.pth'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        self.logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_logger():
    # set up logger
    root_path = os.path.join('log', 'ImageNet')
    if 'evan' in config.MODEL.NAME:
        root_path = os.path.join(root_path, 'evan')
    elif 'pid' in config.MODEL.NAME:
        root_path = os.path.join(root_path, 'pidnet')
    os.makedirs(root_path, exist_ok=True)
    log_file = '{}_{}.log'.format(config.MODEL.NAME, time_str)
    final_log_file = os.path.join(root_path, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

if __name__ == '__main__':
    main()
