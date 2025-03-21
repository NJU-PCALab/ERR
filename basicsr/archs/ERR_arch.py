import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange, repeat
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from typing import Optional, Callable
from timm.models.layers import DropPath

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

from .kan_linear import kan
from .dct_util import *
from .kanformer import KAN

from basicsr.utils.registry import ARCH_REGISTRY


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class BiBranch_Gated_Modulation_Unit(nn.Module):    #
    def __init__(self, dim=32, match_factor=1, bias=True):
        super(BiBranch_Gated_Modulation_Unit, self).__init__()
        self.num_matching = int(dim/match_factor)
        self.fuse1 = nn.Sequential(
                                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0,
                                               bias=bias),
                                    )
        self.fuse2= nn.Sequential(
                                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0,
                                               bias=bias),
                                    )
        self.gate = nn.Sequential(
                                    nn.Conv2d(dim, dim*2, kernel_size=3, stride=1, padding=1,
                                               bias=bias), nn.GELU(),
                                    )
        self.conv = nn.Sequential(
                                    nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0,
                                               bias=bias),
                                    )
    def forward(self, x, perception):
        b, c, h, w = x.size()
        x1=self.fuse1(x)
        p1=self.fuse1(perception)
        f=x1*p1
        gate=self.gate(f) 
        
        gate1, gate2 = torch.chunk(gate, 2, dim=1)
        x=x*gate1+perception
        perception=perception*gate2+x
        output = torch.cat((x, perception), dim=1)
        output =self.conv(output) 
        
        return output


class Matching_transformation(nn.Module):   #
    def __init__(self, dim=32, match_factor=1, ffn_expansion_factor=2, scale_factor=8, bias=True):
        super(Matching_transformation, self).__init__()
        
        self.num_matching = int(dim)
        self.channel = dim
        hidden_features = int(self.channel * ffn_expansion_factor)
        self.matching = BiBranch_Gated_Modulation_Unit(dim=dim, match_factor=match_factor)

        self.perception = nn.Conv2d(3 * dim, dim, 1, bias=bias)

        self.dwconv = nn.Sequential(nn.Conv2d( self.num_matching, hidden_features, 1, bias=bias),
                                    nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                              groups=hidden_features, bias=bias), nn.GELU(),
                                    nn.Conv2d(hidden_features, self.num_matching, 1, bias=bias))
        self.conv12 = nn.Conv2d(self.num_matching, self.channel, 1, bias=bias)

    def forward(self, x, perception):
        perception = self.perception(perception)
        
        filtered_candidate_maps1 = self.matching(x, perception)
        
        dwconv = self.dwconv(filtered_candidate_maps1)
        out = self.conv12(dwconv * filtered_candidate_maps1)+x

        return out


class FeedForward(nn.Module):   #
    def __init__(self, dim=32, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True, ffn_matching=True):
        super(FeedForward, self).__init__()
        self.num_matching = int(dim/match_factor)
        self.channel = dim
        self.matching = ffn_matching
        hidden_features = int(self.channel * ffn_expansion_factor)

        self.project_in = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, 1, bias=bias),
            nn.Conv2d(hidden_features, self.channel, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias)
        )
        if self.matching is True:
            self.matching_transformation = Matching_transformation(dim=dim,
                                                                   match_factor=match_factor,
                                                                   ffn_expansion_factor=ffn_expansion_factor,
                                                                   scale_factor=scale_factor,
                                                                   bias=bias)

        self.project_out = nn.Sequential(
            nn.Conv2d(self.channel, hidden_features, kernel_size=3, stride=1, padding=1, groups=self.channel, bias=bias),
            # nn.GELU(),
            nn.Conv2d(hidden_features, self.channel, 1, bias=bias))

    def forward(self, x, perception):
        project_in = self.project_in(x)
        if perception is not None:
            out = self.matching_transformation(project_in, perception)
        else:
            out = project_in
        project_out = self.project_out(out)
        return project_out


class Attention(nn.Module): #
    def __init__(self, dim, num_heads, match_factor=2,ffn_expansion_factor=2,scale_factor=8, bias=True, attention_matching=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.matching = attention_matching
        if self.matching is True:
            self.matching_transformation = Matching_transformation(dim=dim,
                                                                   match_factor=match_factor,
                                                                   ffn_expansion_factor=ffn_expansion_factor,
                                                                   scale_factor=scale_factor,
                                                                   bias=bias)

    def forward(self, x, perception):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        if self.matching is True:
            q = self.matching_transformation(q, perception)
            k = self.matching_transformation(k, perception)
            v = self.matching_transformation(v, perception)
        else:
            q = q
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward_Restormer(nn.Module): #
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_Restormer, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Downsample(nn.Module):    #
    def __init__(self, n_feat,scale):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.body(x)
    

class Embedding1(nn.Module):    #
    def __init__(self, n_feat):
        scale=2
        super(Embedding1, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.body(x)


class Embedding2(nn.Module):    #
    def __init__(self, n_feat):
        super(Embedding2, self).__init__()
        scale=2

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):  #
    def __init__(self, n_feat,scale):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.body(x)


class ConvBlock(nn.Module): #
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1
        )  # depthwise conv
        
        self.pwconv1 = nn.Linear(
            dim, dim
        )
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class TransformerBlock(nn.Module):  #
    def __init__(self, dim=32, num_heads=1, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_matching=True, ffn_matching=True, ffn_restormer=False):
        super(TransformerBlock, self).__init__()
        self.dim =dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim=dim,
                              num_heads=num_heads,
                              match_factor=match_factor,
                              ffn_expansion_factor=ffn_expansion_factor,
                              scale_factor=scale_factor,
                              bias=bias,
                              attention_matching=attention_matching)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_restormer = ffn_restormer
        if self.ffn_restormer is False:
            self.ffn = FeedForward(dim=dim,
                                   match_factor=match_factor,
                                   ffn_expansion_factor=ffn_expansion_factor,
                                   scale_factor=scale_factor,
                                   bias=bias,
                                   ffn_matching=ffn_matching)
        else:
            self.ffn = FeedForward_Restormer(dim=dim,
                                             ffn_expansion_factor=ffn_expansion_factor,
                                             bias=bias)
        self.LayerNorm = LayerNorm(dim * 3)

    def forward(self, x, perception):
        percetion = self.LayerNorm(perception)
        x = x + self.attn(self.norm1(x), percetion)
        if self.ffn_restormer is False:
            x = x + self.ffn(self.norm2(x), percetion)
        else:
            x = x + self.ffn(self.norm2(x))
        return x


class ResBlock_TransformerBlock(nn.Module): #
    """
    Use preactivation version of residual block, the same as taming
    """

    def __init__(self, dim=32, num_heads=1, match_factor=2, ffn_expansion_factor=2, scale_factor=8, bias=True,
                 LayerNorm_type='WithBias', attention_matching=True, ffn_matching=True, ffn_restormer=False, unit_num=3):
        super(ResBlock_TransformerBlock, self).__init__()
        self.unit_num = unit_num
        self.TransformerBlock = nn.ModuleList()

        for i in range(self.unit_num):
            self.TransformerBlock.append(TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                match_factor=match_factor,
                ffn_expansion_factor=ffn_expansion_factor,
                scale_factor=scale_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                attention_matching=attention_matching,
                ffn_matching=ffn_matching,
                ffn_restormer=ffn_restormer))

    def forward(self, input, perception):
        tmp = input
        for i in range(self.unit_num):
            tmp = self.TransformerBlock[i](tmp, perception)

        out = 0.2 * tmp + input
        return out


class Perception_fusion(nn.Module): #
    def __init__(self, dim=32):
        super(Perception_fusion, self).__init__()
        self.channel = dim
        self.conv11 = nn.Conv2d(3 * self.channel, 3 * self.channel, 1, 1)
        self.dwconv = nn.Conv2d(3 * self.channel, 6 * self.channel, kernel_size=3, stride=1, padding=1,
                                groups=3 * self.channel)
    def forward(self, feature1, feature2, feature3):
        concat = torch.cat([feature1, feature2, feature3], dim=1)
        conv11 = self.conv11(concat)
        dwconv1, dwconv2 = self.dwconv(conv11).chunk(2, dim=1)
        b, c, h, w = dwconv1.size()
        dwconv1 = dwconv1.flatten(2, 3)
        dwconv1 = F.softmax(dwconv1, dim=1)
        dwconv1 = dwconv1.reshape(b, c, h, w)
        perception = torch.mul(dwconv1, concat) + dwconv2
        return perception


class SimpleGate(nn.Module):    #
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class ffn(nn.Module):   #
    def __init__(self, num_feat, ffn_expand=2):
        super(ffn, self).__init__()

        dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel//2, num_feat, kernel_size=1, padding=0, stride=1)
        
        self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1)*x2
        # x = x * self.sca(x)
        x = self.conv3(x)
        return x


class EMBEDD_IN(nn.Module):   #
    def __init__(self, n_feat,scale):
        super( EMBEDD_IN, self).__init__()

        self.embedding = nn.Sequential(nn.Conv2d(n_feat, n_feat*scale, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.embedding(x)


class EMBEDD_OUT(nn.Module): #
    def __init__(self, n_feat,scale):
        super(EMBEDD_OUT, self).__init__()

        self.embedding = nn.Sequential(nn.Conv2d(n_feat*scale, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.embedding(x)

class SS2D(nn.Module):  #
    def __init__(
            self,
            d_model,
            d_state=16,
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
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
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

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

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

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LFSSBlock(nn.Module): #
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x)) 
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x
    


    

class mambablock(nn.Module):    #
    def __init__(self, dim, n_l_blocks=1, n_h_blocks=1, expand=2):
        super().__init__()
        scale_factor=2
        self.embedding_out = EMBEDD_OUT(dim,scale_factor)
        self.embedding_in = EMBEDD_IN(dim,scale_factor)
        self.l_blk = nn.Sequential(*[LFSSBlock(dim*scale_factor, expand=expand) for _ in range(n_l_blocks)])

    
    def forward(self, x):
        x=self.embedding_in(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        for l_layer in self.l_blk:
            x = l_layer(x, [h, w])
        x= rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x=self.embedding_out(x)
        
        return x
    

class MambaNet(nn.Module):  #
    def __init__(self, in_chn=3, wf=48, n_l_blocks=[1,2,2], n_h_blocks=[1,1,1], ffn_scale=2):
        super(MambaNet, self).__init__()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        scale_factor=2
        # encoder of UNet-64
        prev_channels = 0
        self.down_group1 = mambablock(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)
        self.downsample1 = Downsample(wf,scale_factor)
        self.down_group2 = mambablock(wf, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)

        self.up_group1 = mambablock(wf, n_l_blocks=n_l_blocks[0], n_h_blocks=n_h_blocks[0], expand=ffn_scale)
        self.upsample1 = Upsample(wf,scale_factor)
        self.up_group2 = mambablock(wf, n_l_blocks=n_l_blocks[1], n_h_blocks=n_h_blocks[1], expand=ffn_scale)

        self.last = nn.Conv2d(wf, in_chn, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        img = x
        x = self.conv_01(img)
        x = self.down_group1(x)
        skip1=x 
        x=self.downsample1(x)
        x = self.down_group2(x)
        x=self.up_group1(x)
        x=self.upsample1(x)
        x=skip1+x
        x=self.up_group2(x)
        x = self.last(x) + img

        return x


class AAPLayer(nn.Module):
    def __init__(self,dim, pool_sizes=[1, 2, 4]):
        super(AAPLayer, self).__init__()
        self.pool_sizes = pool_sizes
        self.downsample1 = Downsample(dim,4)
        self.downsample2 = Downsample(dim,2)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        features = []
        height1,width1=  int(height / 2), int(width / 2)
        height2,width2 =  int(height / 4), int(width / 4)
        height3,width3 =  int(height / 8), int(width / 8)
        pool0= nn.AdaptiveAvgPool2d((1,1))
        pooled_x0 = pool0(x)

        poolx1 =self.downsample1(x)
        pool1 = nn.AdaptiveAvgPool2d((height3,width3))
        pooled_x1 = pool1(poolx1)*pooled_x0 +pooled_x0 
        
        poolx2 =self.downsample2(x)
        pool2 = nn.AdaptiveAvgPool2d((height3,width3))
        pooled_x2 = pool2(poolx2)*pooled_x0 +pooled_x0
        
        
        pool3 = nn.AdaptiveAvgPool2d((height3,width3))
        poolx3 = pool3(x)*pooled_x0 +pooled_x0

        pool=torch.cat([pooled_x1, pooled_x2,poolx3], dim=1)
        
        return pool
    

@ARCH_REGISTRY.register()
class ERR(nn.Module):
    def __init__(self, channel_query_dict, number_block, num_heads=8, match_factor=2, ffn_expansion_factor=2, scale_factor=8,scale_factor1=4, bias=True,
                 LayerNorm_type='WithBias', attention_matching=True, ffn_matching=True, ffn_restormer=False,unit_num=3):
        super().__init__()
        self.channel_query_dict = channel_query_dict
        
        self.enter1 = nn.Sequential(nn.Conv2d(3, channel_query_dict[256], 3, 1, 1))
        self.spp = AAPLayer(channel_query_dict[256],pool_sizes=[1, 2, 4])
        self.downsample1 = Downsample(channel_query_dict[256],scale_factor)
        self.number_block = number_block
        self.number_block1 = number_block*2
        self.block = nn.ModuleList()
        for i in range(self.number_block):
            self.block.append(ResBlock_TransformerBlock(dim=channel_query_dict[256],
                                                        num_heads=num_heads,
                                                        match_factor=match_factor,
                                                        ffn_expansion_factor=ffn_expansion_factor,
                                                        scale_factor=scale_factor,
                                                        bias=bias,
                                                        LayerNorm_type=LayerNorm_type,
                                                        attention_matching=attention_matching,
                                                        ffn_matching=ffn_matching,
                                                        ffn_restormer=ffn_restormer,
                                                        unit_num=unit_num))
        self.down = Downsample(channel_query_dict[256],2)
        self.downavg = Downsample(channel_query_dict[256]*3,2)
        self.block1 = nn.ModuleList()
        for i in range(self.number_block1):
            self.block1.append(ResBlock_TransformerBlock(dim=channel_query_dict[256],
                                                        num_heads=num_heads,
                                                        match_factor=match_factor,
                                                        ffn_expansion_factor=ffn_expansion_factor,
                                                        scale_factor=scale_factor,
                                                        bias=bias,
                                                        LayerNorm_type=LayerNorm_type,
                                                        attention_matching=attention_matching,
                                                        ffn_matching=ffn_matching,
                                                        ffn_restormer=ffn_restormer,
                                                        unit_num=unit_num))
        self.up= Upsample(channel_query_dict[256],2)
        self.block2 = nn.ModuleList()
        for i in range(self.number_block):
            self.block2.append(ResBlock_TransformerBlock(dim=channel_query_dict[256],
                                                        num_heads=num_heads,
                                                        match_factor=match_factor,
                                                        ffn_expansion_factor=ffn_expansion_factor,
                                                        scale_factor=scale_factor,
                                                        bias=bias,
                                                        LayerNorm_type=LayerNorm_type,
                                                        attention_matching=attention_matching,
                                                        ffn_matching=ffn_matching,
                                                        ffn_restormer=ffn_restormer,
                                                        unit_num=unit_num))
        self.upsample1 = Upsample(channel_query_dict[256],scale_factor)
        self.outer1 = nn.Sequential(nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))

        dim2=channel_query_dict[256]+3
        self.enter2 = nn.Sequential(nn.Conv2d(dim2, channel_query_dict[256], 3, 1, 1))
        self.downsample2 = Downsample(channel_query_dict[256],scale_factor1)
        
        self.middle = MambaNet(in_chn=channel_query_dict[256], wf=32, n_l_blocks=[2,2,3], n_h_blocks=[2,2,1], ffn_scale=2)
        self.upsample2 = Upsample(channel_query_dict[256],scale_factor1)
        self.outer2 = nn.Sequential(nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))

        dim3=channel_query_dict[256]+6
        self.enter3 = nn.Sequential(nn.Conv2d(dim3, channel_query_dict[256], 3, 1, 1))
        self.deep = ConvBlock(channel_query_dict[256])
        self.embed_in = Embedding1(channel_query_dict[256])
        self.deep1 = ConvBlock(channel_query_dict[256])
        self.embed_out= Embedding2(channel_query_dict[256])
        self.FW_KAN1 = KAN(in_features=channel_query_dict[256], hidden_features=channel_query_dict[256], act_layer=nn.GELU, drop=0., )
        self.FW_KAN2 = KAN(in_features=channel_query_dict[256], hidden_features=channel_query_dict[256], act_layer=nn.GELU, drop=0., )
        self.FW_KAN3 = KAN(in_features=channel_query_dict[256], hidden_features=channel_query_dict[256], act_layer=nn.GELU, drop=0., )
        
        #self.downsample = Downsample(channel_query_dict[256],scale_factor)
        self.upsample = Upsample(channel_query_dict[256],scale_factor)
        self.convf = nn.Conv2d(channel_query_dict[256]*2, channel_query_dict[256], 1)
        self.out = nn.Sequential(ConvBlock(channel_query_dict[256]),
                                 ConvBlock(channel_query_dict[256]),
                                 nn.Conv2d(channel_query_dict[256], 3, 3, 1, 1))
        self.beta = nn.Parameter(torch.ones(channel_query_dict[256], 1, 1))
    
        self.dct = DCT2x_torch()
        self.idct = IDCT2x_torch()

    def forward(self, x):
        ori = x
        
        #Zero Frequency Enhancer start
        enter1 = self.enter1(x)
        xpool=self.spp(enter1)
        x1 = self.downsample1(enter1)
        for i in range(self.number_block):
             shallow1 = self.block[i](x1,xpool)
        skip_shadow=shallow1
        shallow1=self.down(shallow1)
        xpool_d1=self.downavg(xpool)
        for i in range(self.number_block):
             shallow1 = self.block1[i](shallow1,xpool_d1)
        shallow1=self.up(shallow1)
        for i in range(self.number_block):
             shallow1 = self.block2[i](shallow1+skip_shadow,xpool)
        upsample1 = self.upsample(shallow1)
        out1=self.outer1(upsample1)+ori
        #Zero  Frequency Enhancer end
        
        #Low Frequency Restorer start
        xx=torch.cat([x, upsample1], dim=1)
        enter2 = self.enter2(xx)
        x2 = self.downsample2(enter2)
        middle = self.middle(x2)
        upsample2 = self.upsample2(middle)
        out2=self.outer2(upsample2)+out1
        xxx=torch.cat([x, out1,upsample2], dim=1)
        #Low Frequency Restorer end

        #High Frequency Restorer start
        enter3 = self.enter3(xxx)
        enter3 = self.deep(enter3)
        enter3_k1=enter3
        enter3 = self.embed_in(enter3)
        deep = self.deep1(enter3)
        deep_k1=deep

        B, C, H, W = deep.shape
        deep_freq = self.dct(deep)
        
        window_size = 64

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            deep_freq = F.pad(deep_freq, (0, pad_w, 0, pad_h), mode='reflect')
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        num_patches_H = H_padded // window_size
        num_patches_W = W_padded // window_size

        processed_deep_freq = torch.zeros_like(deep_freq)

        for i in range(num_patches_H):
            for j in range(num_patches_W):
                h_start = i * window_size
                h_end = h_start + window_size
                w_start = j * window_size
                w_end = w_start + window_size

                window = deep_freq[:, :, h_start:h_end, w_start:w_end]  # [B, C, window_size, window_size]

                B_window = window.size(0)
                window = window.contiguous().view(B_window, C, -1).permute(0, 2, 1)  # [B, N_window, C]

                window =self.FW_KAN1(window, window_size, window_size)
                window =self.FW_KAN2(window, window_size, window_size)+window
                window =self.FW_KAN3(window, window_size, window_size)+window

                window = window.permute(0, 2, 1).contiguous().view(B_window, C, window_size, window_size)

                processed_deep_freq[:, :, h_start:h_end, w_start:w_end] = window

        if pad_h > 0 or pad_w > 0:
            processed_deep_freq = processed_deep_freq[:, :, :H, :W]

        deep = self.idct(processed_deep_freq)
        
        out_emmbed = (1-self.beta)*self.convf(torch.cat([deep, deep_k1], dim=1))+self.beta*deep_k1
        out_emmbed = self.embed_out(out_emmbed)+enter3_k1

        out = self.out(out_emmbed) + out2
        #High Frequency Restorer end


        return out, out1,out2
    

    @torch.no_grad()
    def test(self, x):
        ori = x
        
        #Zero Frequency Enhancer start
        enter1 = self.enter1(x)
        xpool=self.spp(enter1)
        x1 = self.downsample1(enter1)
        for i in range(self.number_block):
             shallow1 = self.block[i](x1,xpool)
        skip_shadow=shallow1
        shallow1=self.down(shallow1)
        xpool_d1=self.downavg(xpool)
        for i in range(self.number_block):
             shallow1 = self.block1[i](shallow1,xpool_d1)
        shallow1=self.up(shallow1)
        for i in range(self.number_block):
             shallow1 = self.block2[i](shallow1+skip_shadow,xpool)
        upsample1 = self.upsample(shallow1)
        out1=self.outer1(upsample1)+ori
        #Zero  Frequency Enhancer end
        
        #Low Frequency Restorer start
        xx=torch.cat([x, upsample1], dim=1)
        enter2 = self.enter2(xx)
        x2 = self.downsample2(enter2)
        middle = self.middle(x2)
        upsample2 = self.upsample2(middle)
        out2=self.outer2(upsample2)+out1
        xxx=torch.cat([x, out1,upsample2], dim=1)
        #Low Frequency Restorer end

        #High Frequency Restorer start
        enter3 = self.enter3(xxx)
        enter3 = self.deep(enter3)
        enter3_k1=enter3
        enter3 = self.embed_in(enter3)
        deep = self.deep1(enter3)
        deep_k1=deep

        B, C, H, W = deep.shape
        deep_freq = self.dct(deep)
        
        window_size = 64

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            deep_freq = F.pad(deep_freq, (0, pad_w, 0, pad_h), mode='reflect')
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        num_patches_H = H_padded // window_size
        num_patches_W = W_padded // window_size

        processed_deep_freq = torch.zeros_like(deep_freq)

        for i in range(num_patches_H):
            for j in range(num_patches_W):
                h_start = i * window_size
                h_end = h_start + window_size
                w_start = j * window_size
                w_end = w_start + window_size

                window = deep_freq[:, :, h_start:h_end, w_start:w_end]  # [B, C, window_size, window_size]

                B_window = window.size(0)
                window = window.contiguous().view(B_window, C, -1).permute(0, 2, 1)  # [B, N_window, C]

                window =self.FW_KAN1(window, window_size, window_size)
                window =self.FW_KAN2(window, window_size, window_size)+window
                window =self.FW_KAN3(window, window_size, window_size)+window

                window = window.permute(0, 2, 1).contiguous().view(B_window, C, window_size, window_size)

                processed_deep_freq[:, :, h_start:h_end, w_start:w_end] = window

        if pad_h > 0 or pad_w > 0:
            processed_deep_freq = processed_deep_freq[:, :, :H, :W]

        deep = self.idct(processed_deep_freq)
        
        out_emmbed = (1-self.beta)*self.convf(torch.cat([deep, deep_k1], dim=1))+self.beta*deep_k1
        out_emmbed = self.embed_out(out_emmbed)+enter3_k1

        out = self.out(out_emmbed) + out2
        #High Frequency Restorer end


        return out, out1,out2
    