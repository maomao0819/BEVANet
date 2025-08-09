# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
import torch.optim
from PIL import Image
from configs import config
from configs import update_config

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

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

    parser.add_argument('--p', help='dir for pretrained model', default='pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)
    parser.add_argument('--c', help='number of classes', type=int, default=19)
    args = parser.parse_args()
    update_config(config, args)

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs/'
    
    device = torch.device('cuda')
    model = models.BEVANet.get_model(config, pretrain_path=args.p, task='seg', num_classes=args.c, is_trainning=False)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for img_path in images_list:
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)

            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            sv_img.save(sv_path+img_name)
            print(f"save at {sv_path+img_name}")