# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import argparse
import torch
import torch.nn as nn
import torch.optim
import _init_paths
from configs import config
from configs import update_config
from utils.utils import calculate_FPS
import models

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/evan/evan64mb.yaml",
                        type=str)   
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--c', help='number of classes', type=int, default=19)
    parser.add_argument('--r', help='input resolution', type=int, nargs='+', default=(1024,2048)) 
    args = parser.parse_args()
    update_config(config, args)

    return args

if __name__ == '__main__':
    args = parse_args()
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)
    device = torch.device('cuda')
    model = models.BEVANet.get_model(config, task='seg', num_classes=args.c, is_trainning=False)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, args.r[0], args.r[1]).cuda()
    print('=========Speed Testing=========')
    with torch.no_grad():
        print('FPS: ' + str(calculate_FPS(input, model)))