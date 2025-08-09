# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

class SimpleModel(nn.Module):

  def __init__(self, model, sem_loss):
    super(SimpleModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, boundary_loss, *args, **kwargs):
    
    outputs = self.model(inputs, *args, **kwargs)
    
    h, w = labels.size(1), labels.size(2)
    outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    acc  = self.pixel_acc(outputs, labels)
    loss = self.sem_loss(outputs, labels)

    return torch.unsqueeze(loss,0), outputs, acc, [loss, loss]
  
class FullModel(nn.Module):

  def __init__(self, model, sem_loss, boundary_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.boundary_loss = boundary_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, boundary_gt, *args, **kwargs):
    
    outputs = self.model(inputs, *args, **kwargs)
    
    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    acc  = self.pixel_acc(outputs[-2], labels)
    loss_s = self.sem_loss(outputs[:-1], labels)
    loss_b = self.boundary_loss(outputs[-1], boundary_gt)

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:])>0.8, labels, filler)
    loss_sb = self.sem_loss(outputs[-2], bd_label)
    loss = loss_s + loss_b + loss_sb

    return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.ENV.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    final_output_dir = root_output_dir / dataset / cfg_name / time_str

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.ENV.LOG_DIR) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int_)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

def benchmark(model, input_tensor, iterations=20):
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_tensor)
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_tensor)
        end_time = time.time()
    return (end_time - start_time) / iterations

def calculate_FPS(
    input_tensor=torch.randn(1, 3, 1024, 2048), 
    model=None, 
    device=torch.device('cuda'), 
    iterations = None
):
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input_tensor)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
        
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input_tensor)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    return FPS