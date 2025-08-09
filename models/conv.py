# ------------------------------------------------------------------------------
# Modified based on https://github.com/AlanLi1997/slim-neck-by-gsconv
# Modified based on https://github.com/JierunChen/FasterNet/blob/master/models/fasternet.py
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels=128, out_channels=None, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, bn=nn.BatchNorm2d, act=nn.ReLU):
        super(ConvBNAct, self).__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, groups=groups)
        self.bn = bn(out_channels)
        self.act = act()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    

class DWConv(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, dilation, bias=bias, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class PWConv(nn.Module):
    def __init__(self, in_channels=768, out_channels=None, bias=True):
        super(PWConv, self).__init__()
        out_channels = out_channels or in_channels
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.pwconv(x)
        return x
    

class DPWConv(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(DWConv, self).__init__()
        self.dwconv = DWConv(dim, kernel_size, stride, padding, dilation, bias=bias)
        self.pwconv = PWConv(dim, bias=bias)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
        return x
    

class GSPWConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, bias=False, channel_attention=False, non_linear=False, shuffle=False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.shuffle = shuffle
        half_channels = int(out_channels // 2)
        if non_linear:
            self.conv1 = ConvBNAct(in_channels, half_channels, kernel_size=1, padding=0, bias=bias, bn=nn.BatchNorm2d, act=nn.Mish)
        else:
            self.conv1 = PWConv(in_channels, half_channels, bias=bias)
        self.conv2 = DWConv(half_channels, kernel_size=1, padding=0, bias=bias)
        self.channel_attention = PWConv(half_channels, bias=bias) if channel_attention else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat((x, self.channel_attention(self.conv2(x))), 1)
        if self.shuffle:
            b, c, h, w = x.data.size()
            b_n = b * c // 2
            x = x.reshape(b_n, 2, h * w)
            x = x.permute(1, 0, 2)
            x = x.reshape(2, -1, c // 2, h, w)
            x = torch.cat((x[0], x[1]), 1)
        return x
    

class GSConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, channel_attention=False, non_linear=True, shuffle=True):
        super().__init__()
        self.shuffle = shuffle
        out_channels = out_channels or in_channels
        half_channels = out_channels // 2
        if non_linear:
            self.conv1 = ConvBNAct(in_channels, half_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias, bn=nn.BatchNorm2d, act=nn.Mish)
            self.conv2 = ConvBNAct(half_channels, half_channels, 5, 1, 2, groups=half_channels, bias=bias, bn=nn.BatchNorm2d, act=nn.Mish)
        else:
            self.conv1 = nn.Conv2d(in_channels, half_channels, kernel_size, stride, padding, dilation, groups=groups, bias=bias)
            self.conv2 = DWConv(half_channels, 5, 1, 2, bias=bias)
        self.channel_attention = PWConv(half_channels, bias=bias) if channel_attention else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat((x, self.channel_attention(self.conv2(x))), 1)
        if self.shuffle:
            b, c, h, w = x.data.size()
            b_n = b * c // 2
            x = x.reshape(b_n, 2, h * w)
            x = x.permute(1, 0, 2)
            x = x.reshape(2, -1, c // 2, h, w)
            x = torch.cat((x[0], x[1]), 1)
        return x


class GSConvns(GSConv):
    # GSConv with a normative-shuffle 
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.shuf = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=False)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat((x1, self.conv2(x1)), 1)
        # normative-shuffle, TRT supported
        out = self.act(self.shuf(x2))
        return out


class PConv(nn.Module):
    def __init__(self, in_channels=768, out_channels=None, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, touch_ratio=0.75, forward='split_cat'):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels_touched = int(in_channels * touch_ratio)
        self.channels_untouched = in_channels - self.in_channels_touched
        if out_channels <= self.channels_untouched:
            self.channels_untouched = 0
        self.out_channels_touched = out_channels - self.channels_untouched
        self.partial_conv = nn.Conv2d(self.in_channels_touched, self.out_channels_touched, kernel_size, stride, padding, dilation, bias=bias)

        if in_channels == out_channels and forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.in_channels_touched, :, :] = self.partial_conv(x[:, :self.in_channels_touched, :, :])
        return x

    def forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.in_channels_touched, self.channels_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        return x


class PDWConv(PConv):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, touch_ratio=0.75, forward='split_cat'):
        super().__init__(in_channels=dim, touch_ratio=touch_ratio, forward=forward)
        self.partial_conv = DWConv(self.dim_touched, kernel_size, stride, padding, dilation, bias=bias)


class PDPWConv(PConv):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, touch_ratio=0.75, forward='split_cat'):
        super().__init__(in_channels=dim, touch_ratio=touch_ratio, forward=forward)
        self.partial_conv = DPWConv(self.dim_touched, kernel_size, stride, padding, dilation, bias=bias)


class PGSPWConv(PConv):
    def __init__(self, in_channels=768, out_channels=None, bias=False, channel_attention=False, non_linear=True, shuffle=True, touch_ratio=0.75, forward='split_cat'):
        super().__init__(in_channels=in_channels, out_channels=out_channels, touch_ratio=touch_ratio, forward=forward)
        self.partial_conv = GSPWConv(self.in_channels_touched, self.out_channels_touched, bias=bias, channel_attention=channel_attention, non_linear=non_linear, shuffle=shuffle)    


class PGSConv(PConv):
    def __init__(self, in_channels=768, out_channels=None, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, channel_attention=False, non_linear=True, shuffle=True, touch_ratio=0.75, forward='split_cat'):
        super().__init__(in_channels=in_channels, out_channels=out_channels, touch_ratio=touch_ratio, forward=forward)
        self.partial_conv = GSConv(self.in_channels_touched, self.out_channels_touched, kernel_size, stride, padding, dilation, bias=bias, channel_attention=channel_attention, non_linear=non_linear, shuffle=shuffle)    


if __name__ == "__main__":
    x = torch.rand(8, 256, 128, 64).to("cuda")
    gsconv = GSConv(256, 128, channel_attention=True).to("cuda")
    print(gsconv(x).size())
    gsconv = GSConvns(256, 128).to("cuda")
    print(gsconv(x).size())
    gspwconv = GSPWConv(in_channels=256).to("cuda")
    print(gspwconv(x).size())
    pgspwconv = PGSPWConv(in_channels=256).to("cuda")
    print(pgspwconv(x).size())
    pwconv = PGSConv(in_channels=256).to("cuda")
    print(pwconv(x).size())
    pgspwconv = PGSPWConv(in_channels=512).to("cuda")
    print(pgspwconv(x).size())