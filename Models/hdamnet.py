from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
from thop import clever_format, profile
from torchsummary import summary
from Models.hdamamba_block import HDAMBlock, PatchMerging2D, PatchEmbed2D, PatchExpand

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from torchsummary import summary#模型可视化

# logger = logging.getLogger(__name__)



import time
import math
import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1 # type: ignore
    from selective_scan import selective_scan_ref as selective_scan_ref_v1 # type: ignore
except:
    pass

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, channels, ratio=16):
        super(SE, self).__init__()
        # 使用 3D 全局平均池化，适用于 5 维输入
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # 两个全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _ = x.size() 
        avg = self.avgpool(x).view(b, c)
        y = self.fc(avg).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class LayerAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(LayerAttentionModule, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=2, stride=2)
        self.se_attention = SE(in_channels, ratio)
    def forward(self, x1, x2):
        # x1: (B, C, H, W)
        # x2: (B, 2C, H/2, W/2)
        x2_up = self.upsample(x2)  
        x = torch.stack([x1, x2_up], dim=0)  
        x = x.permute(1, 2, 3, 4, 0)  
        x = self.se_attention(x)  
        x = x.permute(4, 0, 1, 2, 3)  
        x1_att, x2_att = x[0], x[1] 
        output = x1_att + x2_att     
        return output
    
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        ) 


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class VssblockdownModule(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim,dim):
        super(VssblockdownModule, self).__init__()
        self.vss_block = CloudMambaBlock(hidden_dim=in_channels)
        # self.down = Down(in_channels, out_channels)
        self.downsample = PatchMerging2D(dim)
    def forward(self, x):
  
        
        x = x.permute(0, 2, 3, 1).contiguous()
        vssblock_output = self.vss_block(x)
        vssblock_output = self.downsample(vssblock_output)
        vssblock_output = vssblock_output.permute(0, 3, 1, 2).contiguous()
       
        
        

        # output = torch.concat((vssblock_output, down_output), dim=1)
        # output = vssblock_output + down_output
        return vssblock_output

class VssblockdownModule4(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim,dim):
        super(VssblockdownModule4, self).__init__()
        self.vss_block = CloudMambaBlock(hidden_dim=in_channels)

        self.downsample = PatchMerging2D(dim)
    def forward(self, x):
 
        x = x.permute(0, 2, 3, 1).contiguous()
        vssblock_output = self.vss_block(x)
        vssblock_output = vssblock_output.permute(0, 3, 1, 2).contiguous()
       
        

        return vssblock_output






class CloudMamba(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(CloudMamba, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        # self.down1 = Down(base_c, base_c * 2)
        # self.down2 = Down(base_c * 2, base_c * 4)
        # self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.vssblockdown1 = VssblockdownModule(in_channels=64,out_channels=128, hidden_dim=64,dim=64)
        self.vssblockdown2 = VssblockdownModule(in_channels=128,out_channels=256, hidden_dim=128,dim=128)
        self.vssblockdown3 = VssblockdownModule(in_channels=256,out_channels=512, hidden_dim=256,dim=256)
        self.skip1 = LayerAttentionModule(64)
        self.skip2 = LayerAttentionModule(128)
        self.skip3 = LayerAttentionModule(256)
        # self.vssblockdown4 = VssblockdownModule4(in_channels=512,out_channels=512, hidden_dim=512,dim=512)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        # print('inconv:',x1.shape)
        x2 = self.vssblockdown1(x1)
        # print('down1:',x2.shape)
        x3 = self.vssblockdown2(x2)
        # print('down2:',x3.shape)
        x4 = self.vssblockdown3(x3)
        # print('down3:',x4.shape)
        x5 = self.down4(x4)
        # print('down4:',x5.shape)
        
        # x1 = self.skip1(x1, x2)
        # x2 = self.skip2(x2, x3)
        # x3 = self.skip3(x3, x4)
        x = self.up1(x5, x4)
        x = self.up2(x, self.skip3(x3, x4))
        x = self.up3(x, self.skip2(x2, x3))
        x = self.up4(x, self.skip1(x1, x2))
        logits = self.out_conv(x)

        logits = torch.sigmoid(logits)
        return logits

if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = CloudMamba(3, 1).cuda()
    x = torch.randn(1,3,512,512).cuda()

    start_time = time.time()

    y=model(x)
    print('x输入的shape:',x.shape)
    print('x输出的shape:',y.shape)
    end_time = time.time()

    print(f"运行时间：{end_time - start_time} 秒")
    f, m = profile(model, inputs=(x,))
    f, m = clever_format([f, m], "%.3f")
    print(f, m)
    
    # summary(model, input_size=(3, 384, 384), device='cuda')