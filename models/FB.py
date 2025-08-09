# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, bias=False)                  
        )

        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)
    

class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), 
            BatchNorm(out_channels)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels)
        )
        
    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        
        return p_add + i_add


class BGAF(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.conv_p = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.conv_i = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

        self.out = nn.Sequential(
            BatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):
        p = self.conv_p(p)
        i = self.conv_i(i)
        edge_att = torch.sigmoid(d)
        sum_feature = p + i
        feature = edge_att * p + (1 - edge_att) * i + sum_feature
        return self.out(feature)


if __name__ == "__main__":
    xds = torch.rand(8, 256, 128, 256).to("cuda")
    xs = torch.rand(8, 256, 128, 256).to("cuda")
    xdb = torch.rand(8, 256, 128, 256).to("cuda")
    dfm = BGAF(256, 256, use_conv=True, sum_with_boundary=True).to("cuda")
    print(dfm(xds, xs, xdb).size())