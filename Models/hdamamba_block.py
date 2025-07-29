from torch.nn.modules.utils import _pair
from scipy import ndimage
from torchsummary import summary  # 模型可视化

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
from thop import clever_format, profile 

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1  # type: ignore
    from selective_scan import selective_scan_ref as selective_scan_ref_v1  # type: ignore
except:
    pass

class MultiScaleSEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MultiScaleSEAttention, self).__init__()
        self.channel = channel
        self.reduction = reduction

        self.scale_se = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, max(channel // reduction, 1), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(channel // reduction, 1), channel, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for _ in range(3)
        ])

        self.semantic_att = nn.Sequential(
            nn.Conv2d(3 * channel, 3 * channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(3 * channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(3 * channel, 3 * channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, final_features):

        scale_features = []
        for i, x in enumerate(final_features):
            x = x.permute(0, 3, 1, 2).contiguous()

            y = self.scale_se[i][0](x) 
            y = y.view(y.size(0), -1, 1, 1)
            y = self.scale_se[i][1:](y)

            x = x * y.expand_as(x)
            scale_features.append(x)


        fused_features = torch.cat(scale_features, dim=1)  # (B, 3C, H, W)

 
        att_weights = self.semantic_att(fused_features)  # (B, 3C, H, W)
        fused_features = fused_features * att_weights

        return fused_features

class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        # self.conv = nn.Conv2d(2*dim,dim,kernel_size=3,padding=1)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.conv(x)
        # x = x.permute(0, 2, 3, 1).contiguous()
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            swin_ratio=[0.25, 0.5, 1],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.final_conv2d = nn.ConvTranspose2d(
            in_channels=d_model * 2 * len(swin_ratio),
            out_channels=d_model * 2,
            kernel_size=2,  # 1x1卷积
            stride=2,
            padding=0,
            output_padding=0
        ).to('cuda')
        self.swin_ratio = swin_ratio
        self.attention = MultiScaleSEAttention(d_model*2)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4
 
        swin_ratio = self.swin_ratio  

        processed_xs = []

        final_features = []

        for ratio in swin_ratio:
            crop_h = int(H * ratio)
            crop_w = int(W * ratio)

   
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            x_center = x[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]
            x_topleft = x[:, :, 0:crop_h, 0:crop_w]
            x_topright = x[:, :, 0:crop_h, W - crop_w:W]
            x_bottomleft = x[:, :, H - crop_h:H, 0:crop_w]
            x_bottomright = x[:, :, H - crop_h:H, W - crop_w:W]
            concatenated_features = torch.cat([x_center, x_topleft, x_topright, x_bottomleft, x_bottomright], dim=0)
            H_min, W_min = int(H * swin_ratio[0]), int(W * swin_ratio[0])
            L_min = H_min * W_min 
            group_features_dilationcan_mamba = []
  
            B_i, C_i, H_i, W_i = concatenated_features.shape
            L_i = H_i * W_i

            dilation = max(1, H_i // H_min)  
            
            x_flat = concatenated_features.contiguous().view(B_i, -1, L_i)  
            x_dilated = x_flat[:, :, ::dilation]  
            x_dilated = x_dilated[:, :, :L_min]  
            x_transposed = torch.transpose(concatenated_features, dim0=2, dim1=3).contiguous().view(B_i, -1, L_i)  # (B_i, D_i, L_i)
            x_trans_dilated = x_transposed[:, :, ::dilation] 
            x_trans_dilated = x_trans_dilated[:, :, :L_min]  
            x_hwwh = torch.stack([x_dilated, x_trans_dilated], dim=1).contiguous().view(B_i, 2, -1, L_min)  # (B_i, 2, D_i, L_min)
            x_hwwh_inv = torch.flip(x_hwwh, dims=[-1])  
            xs_i = torch.cat([x_hwwh, x_hwwh_inv], dim=1)  
            processed_xs.append(xs_i)
            B = xs_i.shape[0] 
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_i.view(B, K, -1, L_min), self.x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L_min), self.dt_projs_weight)

            xs_i = xs_i.float().view(B, -1, L_min) 
            dts = dts.contiguous().float().view(B, -1, L_min)  
            Bs = Bs.float().view(B, K, -1, L_min)  
            Cs = Cs.float().view(B, K, -1, L_min)  

            Ds = self.Ds.float().view(-1) 
            As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  
            dt_projs_bias = self.dt_projs_bias.float().view(-1)  

            out_y = self.selective_scan(
                xs_i, dts,
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, K, -1, L_min)
            assert out_y.dtype == torch.float

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L_min)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W_min, H_min), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                            L_min)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W_min, H_min), dim0=2, dim1=3).contiguous().view(B, -1,
                                                                                                               L_min)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H_min, W_min, -1)
            y = self.out_norm(y).to(x.dtype)
            B = int(B // 5)
            group_features_dilationcan_mamba = list(torch.split(y, B, dim=0))
            fused_features = torch.zeros((B, int(H * 0.5), int(W * 0.5), C), device=x.device)
            weight_map = torch.zeros((B, int(H * 0.5), int(W * 0.5), 1), device=x.device)

            positions = ['center', 'topleft', 'topright', 'bottomleft', 'bottomright']

            h_feat, w_feat = group_features_dilationcan_mamba[0].shape[1], group_features_dilationcan_mamba[0].shape[2]

            crop_h = h_feat
            crop_w = w_feat

            coords = {
                'center': ((int(H * 0.5) - crop_h) // 2, (int(W * 0.5) - crop_w) // 2),
                'topleft': (0, 0),
                'topright': (0, int(W * 0.5) - crop_w),
                'bottomleft': (int(H * 0.5) - crop_h, 0),
                'bottomright': (int(H * 0.5) - crop_h, int(W * 0.5) - crop_w)
            }
            for i, pos in enumerate(positions):
                x_i = group_features_dilationcan_mamba[i]
                start_h, start_w = coords[pos]
                end_h, end_w = start_h + h_feat, start_w + w_feat

                fused_features[:, start_h:end_h, start_w:end_w, :] += x_i

                weight_map[:, start_h:end_h, start_w:end_w, :] += 1

            weight_map[weight_map == 0] = 1 

            final_features.append(fused_features / weight_map) 


        # final_fused_features = torch.cat(final_features, dim=-1)
        final_fused_features = self.attention(final_features)
        # final_fused_features = final_fused_features.permute(0, 3, 1, 2).contiguous()
        
        final_fused_features = self.final_conv2d(final_fused_features)
        final_fused_features = final_fused_features.permute(0, 2, 3, 1).contiguous()
        final_fused_features = x.permute(0, 2, 3, 1).contiguous() + final_fused_features  # 残差连接
        final_fused_features = self.out_norm(final_fused_features).to(x.dtype)

        return final_fused_features

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out


class HDAMBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


if __name__ == "__main__":
    # model = VSSBlock(3).cuda()    

    model = SS2D(d_model=3).cuda()  
    x = torch.randn(4, 3, 512, 512).cuda()

    # 记录开始时间
    start_time = time.time()
    # print(model(x))
    print('x输入的shape:', x.shape)
    x = x.permute(0, 2, 3, 1).contiguous() 

    y = model(x)

    y = y.permute(0, 3, 1, 2).contiguous() 
    x = x.permute(0, 3, 1, 2).contiguous() 
    print('x输出的shape:', y.shape)
    end_time = time.time()

    # 计算并打印运行时间
    print(f"运行时间：{end_time - start_time} 秒")
    x = x.permute(0, 2, 3, 1).contiguous() 
    f, m = profile(model, inputs=(x,))
    f, m = clever_format([f, m], "%.3f")
    x = x.permute(0, 3, 1, 2).contiguous()  
    print(f, m)

    # summary(model, input_size=(512, 512, 3), device='cuda')