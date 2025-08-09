# ------------------------------------------------------------------------------
# Modified based on https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .LSKA import LSKA
from .conv import DWConv, PWConv
from .EVAN import SpatialAttention

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(branch_planes * 5, outplanes, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, outplanes, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(branch_planes * 5, outplanes, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, outplanes, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []
        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_out = self.scale_process(torch.cat(scale_list, 1))
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out


class DLKPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DLKPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale5 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.bn = nn.BatchNorm2d(branch_planes, momentum=bn_mom)
        self.lska = SpatialAttention(LSKA, branch_planes, 35, dwconv=DWConv, pwconv=PWConv)
        self.layer_scale_lska = nn.Parameter(1e-2 * torch.ones((branch_planes)), requires_grad=True)
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=2, padding=2, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=2, padding=2, bias=False),
                                    )
        self.process_lska = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process5 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 6, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(branch_planes * 6, outplanes, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, outplanes, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []
        scale0 = self.scale0(x)
        scale_lska = scale0 + self.layer_scale_lska.unsqueeze(-1).unsqueeze(-1) * self.lska(self.bn(scale0))
        x_list.append(scale0)
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process_lska(scale_lska+x_list[2]))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
        x_list.append(self.process5((F.interpolate(self.scale5(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[4])))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class PLKPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PLKPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, branch_planes, bias=False),
                                    )
        self.bn = nn.BatchNorm2d(branch_planes, momentum=bn_mom)
        self.lska = SpatialAttention(LSKA, branch_planes, 23, dwconv=DWConv, pwconv=PWConv)
        self.layer_scale_lska = nn.Parameter(1e-2 * torch.ones((branch_planes)), requires_grad=True)
        self.process = nn.Sequential(
                                    BatchNorm(branch_planes*5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    )
        self.process1 = nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=1, padding=1, groups=1, bias=False)
        self.process2 = nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=1, padding=1, groups=1, bias=False)
        self.process3 = nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=1, padding=1, groups=1, bias=False)
        self.process4 = nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=1, padding=1, groups=1, bias=False)
        self.process_lska = nn.Conv2d(branch_planes, branch_planes, kernel_size=3, dilation=1, padding=1, groups=1, bias=False)

        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 6, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(branch_planes * 6, outplanes, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    PWConv(inplanes, outplanes, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale0 = self.scale0(x)
        scale_lska = scale0 + self.layer_scale_lska.unsqueeze(-1).unsqueeze(-1) * self.lska(self.bn(scale0))
        scale1 = F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+scale0
        scale2 = F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+scale0
        scale3 = F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+scale0
        scale4 = F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+scale0
        scales = self.process(torch.cat([scale1, scale2, scale_lska, scale3, scale4], 1))
        scale1, scale2, scale_lska, scale3, scale4 = torch.chunk(scales, 5, dim=1)
        scale1 = self.process1(scale1)
        scale2 = self.process2(scale2)
        scale3 = self.process3(scale3)
        scale4 = self.process4(scale4)
        scale_lska = self.process_lska(scale_lska)
        out = self.compression(torch.cat([scale0, scale1, scale2, scale3, scale_lska, scale4], 1)) + self.shortcut(x)

        return out


if __name__ == "__main__":
    x = torch.rand(16, 64 * 16, 64, 32).to("cuda")
    inplanes = 64 * 16
    branch_planes = 96
    outplanes = 64 * 4
    dappm = DAPPM(inplanes, branch_planes, outplanes).to("cuda")
    pappm = PAPPM(inplanes, branch_planes, outplanes).to("cuda")
    dalkappm = DLKPPM(inplanes, branch_planes, outplanes, lska=LSKA, pwconv=PWConv).to("cuda")
    palkappm = PLKPPM(inplanes, branch_planes, outplanes, lska=LSKA, pwconv=PWConv).to("cuda")
    print(dappm(x).size())
    print(pappm(x).size())
    print(dalkappm(x).size())
    print(palkappm(x).size())