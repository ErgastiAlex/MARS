
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch import Tensor
from torch.autograd import Variable

from inspect import isfunction
import math
from einops import rearrange, repeat
from functools import partial
# from torch.utils.checkpoint import checkpoint
def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.l = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.l(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features
        )
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)

        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        b, n, k = x.size()
        cls_token, tokens = torch.split(x, [1, n - 1], dim=1)
        x = tokens.reshape(b, int(math.sqrt(n - 1)), int(math.sqrt(n - 1)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False) # h*w x c -> h*w x inner_dim
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False) # 19 x style_dim -> 19 x inner_dim
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False) # 19 x style_dim -> 19 x inner_dim

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, leff=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        if leff:
            self.ff = LocallyEnhancedFeedForward(dim, hidden_features=dim*4, drop=dropout)
        else:
            self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        #x = self.attn1(self.norm1(x)) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth = 8, context_dim=None, leff=False):
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(BasicTransformerBlock(in_channels, n_heads, d_head, context_dim=context_dim, leff=leff))
        

    def forward(self, x, context=None):
        for block in self.blocks:
            x = block(x, context=context)
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, feat_height=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)


        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,use_sigmoid=self.use_sigmoid)
                for d in range(depth)]
        )
            

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        if len(x.size()) == 4:
            b, c, h, w = x.shape
            x = self.norm(x)
            x = self.proj_in(x)
            x = rearrange(x, 'b c h w -> b (h w) c') # bs x h*w x c
            
            for block in self.transformer_blocks:
                x = block(x, context=context)
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.proj_out(x)
        return x + x_in
