# ------------------------------------------------------------------------------
# Modified based on https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from .conv import DWConv, PWConv
from .FS import CKS


class LSKA(nn.Module):
    def __init__(self, dim, kernel_size, dwconv=DWConv, pwconv=PWConv, mode='a'):
        super().__init__()
        self.mode = mode
        if mode == 'h':
            if kernel_size == 7:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,1))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), dilation=2)
            elif kernel_size == 11:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,1))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), dilation=2)
            elif kernel_size == 23:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), dilation=3)
            elif kernel_size == 35:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), dilation=3)
            elif kernel_size == 41:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), dilation=3)
            elif kernel_size == 53:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), dilation=3)
        
        elif mode == 'v':
            if kernel_size == 7:
                self.DW_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(1,0))
                self.DW_D_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), dilation=2)
            elif kernel_size == 11:
                self.DW_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(1,0))
                self.DW_D_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), dilation=2)
            elif kernel_size == 23:
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_v = dwconv(dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), dilation=3)
            elif kernel_size == 35:
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_v = dwconv(dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), dilation=3)
            elif kernel_size == 41:
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_v = dwconv(dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), dilation=3)
            elif kernel_size == 53:
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_v = dwconv(dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), dilation=3)

        else:
            if kernel_size == 7:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,1))
                self.DW_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(1,0))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), dilation=2)
                self.DW_D_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), dilation=2)
            elif kernel_size == 11:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,1))
                self.DW_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(1,0))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), dilation=2)
                self.DW_D_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), dilation=2)
            elif kernel_size == 23:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), dilation=3)
                self.DW_D_conv_v = dwconv(dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), dilation=3)
            elif kernel_size == 35:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), dilation=3)
                self.DW_D_conv_v = dwconv(dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), dilation=3)
            elif kernel_size == 41:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), dilation=3)
                self.DW_D_conv_v = dwconv(dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), dilation=3)
            elif kernel_size == 53:
                self.DW_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2))
                self.DW_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0))
                self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), dilation=3)
                self.DW_D_conv_v = dwconv(dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), dilation=3)
        
        self.pwconv = pwconv(dim)

    def forward(self, x):
        u = x.clone()
        if self.mode == 'h':
            attn = self.DW_conv_h(x)
            attn = self.DW_D_conv_h(attn)
        elif self.mode == 'v':
            attn = self.DW_conv_v(x)
            attn = self.DW_D_conv_v(attn)
        else:
            attn = self.DW_conv_h(x)
            attn = self.DW_conv_v(attn)
            attn = self.DW_D_conv_h(attn)
            attn = self.DW_D_conv_v(attn)
        attn = self.pwconv(attn)
        return u * attn


class SLAK(nn.Module):
    def __init__(self, dim, kernel_size, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        self.conv_small = dwconv(dim, kernel_size=(5, 5), stride=(1,1), padding=(2,2))
        self.conv_big_h = dwconv(dim, kernel_size=(kernel_size, 5), stride=(1,1), padding=(kernel_size//2,2))
        self.conv_big_v = dwconv(dim, kernel_size=(5, kernel_size), stride=(1,1), padding=(2,kernel_size//2))
        self.pwconv = pwconv(dim)

    def forward(self, x):
        u = x.clone()
        attn_small = self.conv_small(x)
        attn_big_h = self.conv_big_h(x)
        attn_big_v = self.conv_big_v(x)
        attn = attn_small + attn_big_h + attn_big_v
        attn = self.pwconv(attn)
        return u * attn


class SDLSKA(nn.Module):
    def __init__(self, dim, kernel_size, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        if kernel_size == 7 or kernel_size == 11:
            self.DW_conv = dwconv(dim, kernel_size=(3, 3), stride=(1,1), padding=(1,1))
        else:
            self.DW_conv = dwconv(dim, kernel_size=(5, 5), stride=(1,1), padding=(2,2))
        if kernel_size == 7:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), dilation=2)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), dilation=2)
        elif kernel_size == 11:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), dilation=2)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), dilation=2)
        elif kernel_size == 23:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), dilation=3)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), dilation=3)
        elif kernel_size == 35:
            self.DW_D_conv_h = dwconv(dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), dilation=3)
            self.DW_D_conv_v = dwconv(dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), dilation=3)
        self.pwconv = pwconv(dim)
        self.FS_kernel = CKS(dim, dwconv=DWConv, pwconv=PWConv)

    def forward(self, x):
        u = x.clone()
        attn_small = self.DW_conv(x)
        attn_big_h = self.DW_D_conv_h(attn_small)
        attn_big_v = self.DW_D_conv_v(attn_small)
        attn = self.FS_kernel(attn_small, attn_big_h, attn_big_v)
        attn = self.pwconv(attn)
        return u * attn
    

if __name__ == "__main__":
    x = torch.rand(8, 16, 64, 64).to("cuda")
    sdlska = SDLSKA(dim=16, kernel_size=35, method='channel_add', fuse_method='softmax').to("cuda")
    print(sdlska(x).size())
    