# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

#environment
_C.ENV = CN()
_C.ENV.OUTPUT_DIR = ''
_C.ENV.LOG_DIR = ''
_C.ENV.GPUS = (0,)
_C.ENV.WORKERS = 4

_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Logging
_C.LOGGING = CN()
_C.LOGGING.PRINT_ITER_FREQ = 20

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'BEVANet'
_C.MODEL.BRANCHES = 3
_C.MODEL.CSP_ARCH = False
_C.MODEL.CSP_KEEP_RATIO = 0.0
_C.MODEL.CSP_CONV_RATIO = 0.0
_C.MODEL.LKA = 'LSKA' # LSKA, SDLSKA, SLAK
_C.MODEL.DWCONV = 'DWConv' # DWConv, PDWConv
_C.MODEL.PWCONV = 'PWConv' # PWConv, GSPWConv, PGSPWConv
_C.MODEL.PPM = 'DAPPM' # DAPPM, PAPPM, DLKPPM, PLKPPM
_C.MODEL.FB = 'Bag' # Bag, Light_Bag, BGAF
_C.MODEL.SEMANTIC_KERNEL_SIZE = [35, 35, 35]
_C.MODEL.DETAIL_KERNEL_SIZE = [23, 23, 23]
_C.MODEL.MLP_EXPAND = [4, 4, 4]
_C.MODEL.NUM_BLOCKS = [2, 2, 3]
_C.MODEL.PLANES = 64
_C.MODEL.PPM_PLANES = 96 
_C.MODEL.HEAD_PLANES = 128
_C.MODEL.PRETRAINED = 'pretrained_models/imagenet/BEVANet_ImageNet.pth'
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 2

_C.LOSS = CN()
_C.LOSS.USE_OHEM = True
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BCE_TYPE = "weight_bce" # weight_bce, annotator_robust
_C.LOSS.BALANCE_WEIGHTS = [0.5, 0.5]
_C.LOSS.SB_WEIGHTS = 0.5

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'data/'
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

# training
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = [1024, 1024]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DETAIL = CN()
_C.TRAIN.DETAIL.MULTI_SCALE = False
_C.TRAIN.DETAIL.SMOOTH_KERNEL = 0
_C.TRAIN.DETAIL.TO_BINARY = True
_C.TRAIN.FLIP = True
_C.TRAIN.RAND_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# validation
_C.VALID = CN()
_C.VALID.VALID_EPOCH_FREQ = 10
_C.VALID.VALID_LAST_EPOCH = 100

# testing
_C.TEST = CN()
_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False

_C.TEST.OUTPUT_INDEX = -1


def update_config(cfg, args):
    cfg.defrost()
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

