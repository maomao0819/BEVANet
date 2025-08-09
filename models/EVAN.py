# ------------------------------------------------------------------------------
# Modified based on https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention
# Modified based on https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/paddleseg/models/rtformer.py
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import math
from .LSKA import SDLSKA
from .conv import DWConv, PWConv


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dwconv=DWConv, pwconv=PWConv, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = pwconv(in_features, hidden_features)
        self.dwconv = dwconv(hidden_features)
        self.act = act_layer()
        self.fc2 = pwconv(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, lka, d_model, kernel_size, dwconv=DWConv, pwconv=PWConv):
        super().__init__()
        self.proj_1 = pwconv(d_model, d_model)
        self.activation = nn.GELU()
        self.spatial_gating_unit = lka(d_model, kernel_size, dwconv=dwconv, pwconv=pwconv)
        self.proj_2 = pwconv(d_model, d_model)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)
    

class EVAN(nn.Module):
    """Implementation of Efficient Visual Attention Network Block."""
    def __init__(self, lka=SDLSKA, dwconv=DWConv, pwconv=PWConv, dim=512, kernel_size=35, mlp_expand=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(lka, dim, kernel_size, dwconv=dwconv, pwconv=pwconv)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_expand)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
    

if __name__ == "__main__":
    x = torch.rand(8, 256, 128, 64).to("cuda")
    lska = EVAN(attn=SpatialAttention, lka=SDLSKA, dim=256, kernel_size=7).to("cuda")
    print(lska(x).size())