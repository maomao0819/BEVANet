# ------------------------------------------------------------------------------
# Modified based on https://github.com/zcablii/LSKNet
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import DWConv, PWConv


class CKS(nn.Module):
    def __init__(self, dim, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        hidden_channel = max(dim//16, 32)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_reduce = nn.Sequential(
            pwconv(dim, hidden_channel),
            dwconv(hidden_channel),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True)
        )
        self.fc_expand_small = nn.Sequential(
            pwconv(hidden_channel, dim),
            dwconv(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.fc_expand_big_h = nn.Sequential(
            pwconv(hidden_channel, dim),
            dwconv(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.fc_expand_big_v = nn.Sequential(
            pwconv(hidden_channel, dim),
            dwconv(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv_squeeze = nn.Conv2d(2, 3, 7, padding=3)

    def norm_weight_channel(self, sig_small, sig_big_h, sig_big_v):
        sig_small = torch.sigmoid(sig_small)
        sig_big_h = torch.sigmoid(sig_big_h)
        sig_big_v = torch.sigmoid(sig_big_v)
        return sig_small, sig_big_h, sig_big_v

    def norm_weight_spatial(self, sig):
        sig = torch.sigmoid(sig)
        sig_small, sig_big_h, sig_big_v = torch.chunk(sig, chunks=3, dim=1)
        return sig_small, sig_big_h, sig_big_v
    
    def forward(self, attn_small, attn_big_h, attn_big_v):
        attn = attn_small + attn_big_h + attn_big_v
        feats_S = self.avg_pool(attn)
        feats_Z = self.fc_reduce(feats_S)
        sig_small_channel = self.fc_expand_small(feats_Z)
        sig_big_h_channel = self.fc_expand_big_h(feats_Z)
        sig_big_v_channel = self.fc_expand_big_v(feats_Z)
        sig_small_channel, sig_big_h_channel, sig_big_v_channel = self.norm_weight_channel(sig_small_channel, sig_big_h_channel, sig_big_v_channel)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg)
        sig_small, sig_big_h, sig_big_v = self.norm_weight_spatial(sig)
        attn = attn_small * sig_small_channel * sig_small + attn_big_h * sig_big_h_channel * sig_big_h + attn_big_v * sig_big_v_channel * sig_big_v
        return attn


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv2d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                BatchNorm(mid_channels)
                                )
        if with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channels, in_channels, 
                                              kernel_size=1, bias=False),
                                    BatchNorm(in_channels)
                                   )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y
        
        return x

if __name__ == "__main__":
    small = torch.rand(4, 16, 64, 64).to("cuda")
    big_h = torch.rand(4, 16, 64, 64).to("cuda")
    big_v = torch.rand(4, 16, 64, 64).to("cuda")
    FS_kernel = CKS(16, method='spatial_extend_norm', fuse_method="softmax").to("cuda")
    output = FS_kernel(small, big_h, big_v)