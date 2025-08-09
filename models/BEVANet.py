# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead
from .FB import Bag
import logging
from .__init__ import module_dict
from .EVAN import EVAN
from .LSKA import LSKA, SDLSKA
from .conv import DWConv, PWConv
from .PPM import DAPPM
from .FS import PagFM
from utils.utils import calculate_FPS

bn_mom = 0.1
algc = False


class BEVANet_backbone(nn.Module):
    def __init__(
        self,
        lka=LSKA,
        dwconv=DWConv,
        pwconv=PWConv,
        semantic_kernel_size=[0, 35, 35],
        detail_kernel_size=[0, 23],
        mlp_expand=[4, 4, 4],
        n_blocks=[2, 2, 4, 4, 2],
        planes=64,
    ):
        super().__init__()
        self.semantic_kernel_size = semantic_kernel_size
        self.detail_kernel_size = detail_kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # STEM
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, n_blocks[0])
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, n_blocks[1], stride=2)

        # Semantic
        if semantic_kernel_size[0] == 0 and n_blocks[2] > 1:
            self.layer3_semantic = self._make_layer(BasicBlock, planes * 2, planes * 4, n_blocks[2], stride=2)
        else:
            self.layer3_down_semantic = self._make_layer(BasicBlock, planes * 2, planes * 4, 1, stride=2)
            self.layer3_semantic = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 4,
                kernel_size=semantic_kernel_size[0],
                mlp_expand=mlp_expand[0]
            )
            self.norm3_semantic = nn.LayerNorm(planes * 4)

        if semantic_kernel_size[1] == 0 and n_blocks[3] > 1:
            self.layer4_semantic = self._make_layer(BasicBlock, planes * 4, planes * 8, n_blocks[3], stride=2)
        else:
            self.layer4_down_semantic = self._make_layer(BasicBlock, planes * 4, planes * 8, 1, stride=2)
            self.layer4_semantic_1 = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 8,
                kernel_size=semantic_kernel_size[1],
                mlp_expand=mlp_expand[1]
            )
            self.norm4_semantic_1 = nn.LayerNorm(planes * 8)
            self.layer4_semantic_2 = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 8,
                kernel_size=semantic_kernel_size[1],
                mlp_expand=mlp_expand[1]
            )
            self.norm4_semantic_2 = nn.LayerNorm(planes * 8)

        if detail_kernel_size[0] == 0:
            self.layer3_detail = self._make_layer(BasicBlock, planes * 2, planes * 2, n_blocks[0])
        else:
            self.layer3_detail_conv = self._make_layer(BasicBlock, planes * 2, planes * 2, 1)
            self.layer3_detail = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 2,
                kernel_size=detail_kernel_size[0],
                mlp_expand=mlp_expand[0]
            )
            self.norm3_detail = nn.LayerNorm(planes * 2)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        self.fuse_sd_3 = PagFM(planes * 2, planes)

        if detail_kernel_size[1] == 0:
            self.layer4_detail = self._make_layer(BasicBlock, planes * 2, planes * 2, n_blocks[1])
        else:
            self.layer4_detail_conv = self._make_layer(BasicBlock, planes * 2, planes * 2, 1)
            self.layer4_detail = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 2,
                kernel_size=detail_kernel_size[1],
                mlp_expand=mlp_expand[1]
            )
            self.norm4_detail = nn.LayerNorm(planes * 2)
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        self.fuse_sd_4 = PagFM(planes * 2, planes)

        self.layer5_detail_semantic = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

    def _make_layer(self, block, inplanes, planes, n_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion

        for i in range(1, n_blocks):
            if i == (n_blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        B, _, H, W = x.size()
        x = self.relu(self.layer1(self.conv1(x)))  # c, 1/4
        x = self.relu(self.layer2(x))  # 2c, 1/8
        if self.semantic_kernel_size[0] == 0:
            xs = self.relu(self.layer3_semantic(x)) # 4c, 1/16
        else:
            xs = self.layer3_semantic(self.layer3_down_semantic(x))  # 4c, 1/16
            xs = self.norm3_semantic(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, -1, int(H / 16), int(W / 16)).contiguous()

        if self.detail_kernel_size[0] != 0:
            xd = self.layer3_detail_conv(x)
            xd = self.layer3_detail(xd)
            xd = self.norm3_detail(xd.flatten(2).transpose(1, 2))
            xd = xd.permute(0, 2, 1).reshape(B, -1, int(H / 8), int(W / 8)).contiguous()
        else:
            xd = self.layer3_detail(x) # 2c, 1/8

        xd = self.fuse_sd_3(xd, self.compression3(xs))

        xds_inter = xd
        if self.semantic_kernel_size[1] == 0:
            xs = self.relu(self.layer4_semantic(xs)) # 4c, 1/16
        else:
            xs = self.layer4_semantic_1(self.layer4_down_semantic(xs))  # 4c, 1/32
            _, C, feat_H, feat_W = xs.size()
            xs = self.norm4_semantic_1(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, C, feat_H, feat_W).contiguous()
            xs = self.layer4_semantic_2(xs)  # 4c, 1/32
            xs = self.norm4_semantic_2(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, C, feat_H, feat_W).contiguous()
        
        if self.detail_kernel_size[0] == 0:
            xd = self.relu(xd)

        if self.detail_kernel_size[1] != 0:
            xd = self.layer4_detail_conv(xd)
            xd = self.layer4_detail(xd) # 2c, 1/8
            xd = self.norm4_detail(xd.flatten(2).transpose(1, 2))
            xd = xd.permute(0, 2, 1).reshape(B, -1, int(H / 8), int(W / 8)).contiguous()
        else:
            xd = self.layer4_detail(xd) # 2c, 1/8
        xd = self.fuse_sd_4(xd, self.compression4(xs))

        if self.detail_kernel_size[1] == 0:
            xd = self.relu(xd)
        xds = self.layer5_detail_semantic(xd)
        return xs, xds, xd, xds_inter


class BEVANet_CLS_32(BEVANet_backbone):
    def __init__(
        self,
        lka=SDLSKA,
        dwconv=DWConv,
        pwconv=PWConv,
        semantic_kernel_size=[35, 35, 35],
        detail_kernel_size=[23, 23],
        mlp_expand=[4, 4],
        n_blocks=[2, 2, 4, 4, 2],
        num_classes=1000,
        planes=64,
        last_planes=2048,
        apply_init=False
    ):
        super().__init__(
            lka=lka,
            dwconv=dwconv,
            pwconv=pwconv,
            semantic_kernel_size=semantic_kernel_size,
            detail_kernel_size=detail_kernel_size, 
            mlp_expand=mlp_expand,
            n_blocks=n_blocks,
            planes=planes
        )
        self.relu = nn.ReLU(inplace=True)
        self.semantic_kernel_size = semantic_kernel_size
        self.last_planes = last_planes
        if semantic_kernel_size[2] == 0 and n_blocks[4] > 1:
            self.layer5_semantic_cls = self._make_layer(Bottleneck, planes * 8, planes * 8, n_blocks[4])
        else:
            self.layer5_down_semantic_cls = self._make_layer(Bottleneck, planes * 8, planes * 8, 1)
            self.layer5_semantic_1_cls = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 16,
                kernel_size=semantic_kernel_size[2],
                mlp_expand=mlp_expand[2]
            )

            self.norm5_semantic_1_cls = nn.LayerNorm(planes * 16)
            self.layer5_semantic_2_cls = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 16,
                kernel_size=semantic_kernel_size[2],
                mlp_expand=mlp_expand[2]
            )
            self.norm5_semantic_2_cls = nn.LayerNorm(planes * 16)
            self.layer5_semantic_bn_cls = self._make_layer(Bottleneck, planes * 16, planes * 8, 1, 1)

        self.layer5_down = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 8, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 8, planes * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 16, momentum=bn_mom),
        ) 
        self.last_layer = nn.Sequential(
            nn.Conv2d(planes * 16, last_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(last_planes, num_classes)

        if apply_init:
            self.apply(self._init_weights)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs, xd, _, _ = super().forward(x)
        if self.semantic_kernel_size[2] == 0:
            xs = self.layer5_semantic_cls(xs) # 4c, 1/16
        else:
            xs = self.layer5_semantic_1_cls(self.layer5_down_semantic_cls(xs))  # 8c, 1/64
            B, _, H, W = xs.size()
            xs = self.norm5_semantic_1_cls(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
            xs = self.layer5_semantic_2_cls(xs)  # 8c, 1/64
            xs = self.norm5_semantic_2_cls(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
            xs = self.layer5_semantic_bn_cls(xs)

        xs = xs + self.layer5_down(xd)
        xs = self.last_layer(self.relu(xs))
        xs = xs.view(-1, self.last_planes)
        xs = self.head(xs)
        return xs


class BEVANet_CLS_64(BEVANet_CLS_32):
    def __init__(
        self,
        lka=SDLSKA,
        dwconv=DWConv,
        pwconv=PWConv,
        semantic_kernel_size=[35, 35, 35],
        detail_kernel_size=[23, 23],
        mlp_expand=[4, 4],
        n_blocks=[2, 2, 4, 4, 2],
        num_classes=1000,
        planes=64,
        last_planes=2048,
        apply_init=False
    ):
        super().__init__(
            lka=lka,
            dwconv=dwconv,
            pwconv=pwconv,
            semantic_kernel_size=semantic_kernel_size,
            detail_kernel_size=detail_kernel_size, 
            mlp_expand=mlp_expand,
            n_blocks=n_blocks,
            num_classes=num_classes,
            planes=planes,
            last_planes=last_planes,
            apply_init=apply_init
        )
        if semantic_kernel_size[2] == 0 and n_blocks[4] > 1:
            self.layer5_semantic_cls = self._make_layer(Bottleneck, planes * 8, planes * 8, n_blocks[4], stride=2)
        else:
            self.layer5_down_semantic_cls = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.layer5_down = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 8, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 8, planes * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 16, momentum=bn_mom),
        )
        if apply_init:
            self.apply(self._init_weights)


class BEVANet_SEG(BEVANet_backbone):
    def __init__(
        self,
        lka=SDLSKA,
        dwconv=DWConv,
        pwconv=PWConv,
        ppm=DAPPM,
        fb=Bag,
        semantic_kernel_size=[35, 35, 35],
        detail_kernel_size=[23, 23],
        mlp_expand=[4, 4],
        n_blocks=[2, 2, 4, 4, 2],
        num_classes=19,
        planes=64,
        ppm_planes=96, 
        head_planes=128,
        seg_augment=True,
        apply_init=False
    ):
        super().__init__(
            lka=lka,
            dwconv=dwconv,
            pwconv=pwconv,
            semantic_kernel_size=semantic_kernel_size,
            detail_kernel_size=detail_kernel_size, 
            mlp_expand=mlp_expand,
            n_blocks=n_blocks,
            planes=planes
        )
        self.semantic_kernel_size = semantic_kernel_size
        self.seg_augment = seg_augment

        if semantic_kernel_size[2] == 0 and n_blocks[4] > 1:
            self.layer5_semantic_seg = self._make_layer(Bottleneck, planes * 8, planes * 8, n_blocks[4], stride=2)
        else:
            self.layer5_down_semantic_seg = self._make_layer(BasicBlock, planes * 8, planes * 16, 1, stride=2)
            self.layer5_semantic_1_seg = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 16,
                kernel_size=semantic_kernel_size[2],
                mlp_expand=mlp_expand[2]
            )
            self.norm5_semantic_1_seg = nn.LayerNorm(planes * 16)
            self.layer5_semantic_2_seg = EVAN(
                lka=lka,
                dwconv=dwconv,
                pwconv=pwconv,
                dim=planes * 16,
                kernel_size=semantic_kernel_size[2],
                mlp_expand=mlp_expand[2]
            )
            self.norm5_semantic_2_seg = nn.LayerNorm(planes * 16)

        self.layer5_detail_boundary = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.spp = ppm(planes * 16, ppm_planes, planes * 4)
        self.dfm = fb(planes * 4, planes * 4)
        # Prediction Head
        if self.seg_augment:
            self.seghead_ds = segmenthead(planes * 2, head_planes, num_classes)
            self.seghead_db = segmenthead(planes * 2, planes, 1) 
        self.head = segmenthead(planes * 4, head_planes, num_classes)

        if apply_init:
            self.apply(self._init_weights)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs, xds, xd, xds_inter = super().forward(x)
        if self.semantic_kernel_size[2] == 0:
            xs = self.layer5_semantic_seg(xs) # 4c, 1/16
        else:
            xs = self.layer5_semantic_1_seg(self.layer5_down_semantic_seg(xs))  # 8c, 1/64
            B, _, H, W = xs.size()
            xs = self.norm5_semantic_1_seg(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
            xs = self.layer5_semantic_2_seg(xs)  # 8c, 1/64
            xs = self.norm5_semantic_2_seg(xs.flatten(2).transpose(1, 2))
            xs = xs.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        xdb = self.layer5_detail_boundary(xd)
        _, _, H, W = xds.size()
        xs = F.interpolate(
            self.spp(xs),
            size=[H, W],
            mode='bilinear', align_corners=algc
        )
        x = self.head(self.dfm(xds, xs, xdb))
        if self.seg_augment:
            x_extra_ds = self.seghead_ds(xds_inter)
            x_extra_db = self.seghead_db(xd)
            return [x_extra_ds, x, x_extra_db]
        else:
            return x
   

def get_seg_model(cfg, num_classes=19, is_trainning=True):
    model_LKA = module_dict["LKA"][cfg.MODEL.LKA]
    model_DWCONV = module_dict["conv"][cfg.MODEL.DWCONV]
    model_PWCONV = module_dict["conv"][cfg.MODEL.PWCONV]
    model_PPM = module_dict["PPM"][cfg.MODEL.PPM]
    model_FB = module_dict["FB"][cfg.MODEL.FB]
    model = BEVANet_SEG(
        lka=model_LKA,
        dwconv=model_DWCONV,
        pwconv=model_PWCONV,
        ppm=model_PPM,
        fb=model_FB,
        semantic_kernel_size=cfg.MODEL.SEMANTIC_KERNEL_SIZE,
        detail_kernel_size=cfg.MODEL.DETAIL_KERNEL_SIZE,
        mlp_expand=cfg.MODEL.MLP_EXPAND,
        n_blocks=cfg.MODEL.NUM_BLOCKS,
        num_classes=num_classes,
        planes=cfg.MODEL.PLANES,
        ppm_planes=cfg.MODEL.PPM_PLANES, 
        head_planes=cfg.MODEL.HEAD_PLANES,
        seg_augment=is_trainning
    )
    return model


def get_cls_model(cfg):
    model_LKA = module_dict["LKA"][cfg.MODEL.LKA]
    model_DWCONV = module_dict["conv"][cfg.MODEL.DWCONV]
    model_PWCONV = module_dict["conv"][cfg.MODEL.PWCONV]
    model = BEVANet_CLS_32(
        lka=model_LKA,
        dwconv=model_DWCONV,
        pwconv=model_PWCONV,
        semantic_kernel_size=cfg.MODEL.SEMANTIC_KERNEL_SIZE,
        detail_kernel_size=cfg.MODEL.DETAIL_KERNEL_SIZE,
        mlp_expand=cfg.MODEL.MLP_EXPAND,
        n_blocks=cfg.MODEL.NUM_BLOCKS,
        num_classes=1000,
        planes=cfg.MODEL.PLANES,
    )
    return model


def get_model(cfg, pretrain_path='', task='cls', num_classes=0, is_trainning=True):
    if num_classes == 0:
        num_classes = cfg.DATASET.NUM_CLASSES
    if pretrain_path == '':
        pretrain_path = cfg.MODEL.PRETRAINED
    imgnet_pretrained = 'imagenet' in pretrain_path or 'ImageNet' in pretrain_path
    model = get_cls_model(cfg) if task == 'cls' else get_seg_model(cfg, num_classes, is_trainning=is_trainning)
    if os.path.exists(pretrain_path):
        pretrained_dict = torch.load(pretrain_path, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        if imgnet_pretrained:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        else:
            pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        model_dict.update(pretrained_dict)
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict=False)
    elif is_trainning == True:
        logging.info('Train from scratch!')
    return model


if __name__ == '__main__':
    
    # Comment batchnorms here and in model_utils before testing speed since the batchnorm could be integrated into conv operation
    # (do not comment all, just the batchnorm following its corresponding conv layer)

    device = torch.device('cuda')
    model = BEVANet_SEG().to("cuda")
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, 1024, 2048).cuda()
    print('=========Speed Testing=========')
    with torch.no_grad():
        print('FPS: ' + str(calculate_FPS(input, model)))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))