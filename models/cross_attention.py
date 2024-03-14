import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import urllib
import warnings

from tqdm import tqdm

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.resnetv2 import ResNetV2
from timm.models.registry import register_model
from torchvision import transforms


def cosine_distance(x1, x2):
    '''
    x1      =  [b, h, n, k]
    x2      =  [b, h, k, m]
    output  =  [b, h, n, m]
    '''
    dots = torch.matmul(x1, x2)
    scale = torch.einsum('bhi, bhj -> bhij', 
            (torch.norm(x1, 2, dim = -1), torch.norm(x2, 2, dim = -2)))
    return (dots / scale)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    #from https://github.com/pprp/timm/blob/master/timm/models/crossvit.py
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,y):
        B, N, C = x.shape
        B, M, C = y.shape
        
        # BMC -> BMH(C/H) -> BHM(C/H)
        q = self.wq(y).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHM(C/H) @ BH(C/H)N -> BHMN
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, M, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attn

class CrossAttentionBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x,y):
        _x,attn =  self.attn(self.norm1(x),self.norm1(y))
        y = y + self.drop_path(_x)
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return y,attn

# class CrossViT(nn.Module):
#     """ Vision Transformer with support for patch or hybrid CNN input stage
#     """

#     def __init__(
#             self, img_size=224, img_scale=(1.0, 1.0), patch_size=(8, 16), in_chans=3, num_classes=1000,
#             embed_dim=(192, 384), depth=((1, 3, 1), (1, 3, 1), (1, 3, 1)), num_heads=(6, 12), mlp_ratio=(2., 2., 4.),
#             multi_conv=False, crop_scale=False, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool='token',
#     ):
#         super().__init__()
#         assert global_pool in ('token', 'avg')

#         self.num_classes = num_classes
#         self.global_pool = global_pool
#         self.img_size = to_2tuple(img_size)
#         img_scale = to_2tuple(img_scale)
#         self.img_size_scaled = [tuple([int(sj * si) for sj in self.img_size]) for si in img_scale]
#         self.crop_scale = crop_scale  # crop instead of interpolate for scale
#         num_patches = _compute_num_patches(self.img_size_scaled, patch_size)
#         self.num_branches = len(patch_size)
#         self.embed_dim = embed_dim
#         self.num_features = sum(embed_dim)
#         self.patch_embed = nn.ModuleList()

#         # hard-coded for torch jit script
#         for i in range(self.num_branches):
#             setattr(self, f'pos_embed_{i}', nn.Parameter(torch.zeros(1, 1 + num_patches[i], embed_dim[i])))
#             setattr(self, f'cls_token_{i}', nn.Parameter(torch.zeros(1, 1, embed_dim[i])))

#         for im_s, p, d in zip(self.img_size_scaled, patch_size, embed_dim):
#             self.patch_embed.append(
#                 PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d, multi_conv=multi_conv))

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         total_depth = sum([sum(x[-2:]) for x in depth])
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
#         dpr_ptr = 0
#         self.blocks = nn.ModuleList()
#         for idx, block_cfg in enumerate(depth):
#             curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
#             dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
#             blk = MultiScaleBlock(
#                 embed_dim, num_patches, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
#             dpr_ptr += curr_depth
#             self.blocks.append(blk)

#         self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])
#         self.head = nn.ModuleList([
#             nn.Linear(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
#             for i in range(self.num_branches)])

#         for i in range(self.num_branches):
#             trunc_normal_(getattr(self, f'pos_embed_{i}'), std=.02)
#             trunc_normal_(getattr(self, f'cls_token_{i}'), std=.02)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         out = set()
#         for i in range(self.num_branches):
#             out.add(f'cls_token_{i}')
#             pe = getattr(self, f'pos_embed_{i}', None)
#             if pe is not None and pe.requires_grad:
#                 out.add(f'pos_embed_{i}')
#         return out

#     @torch.jit.ignore
#     def group_matcher(self, coarse=False):
#         return dict(
#             stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
#             blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
#         )

#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         assert not enable, 'gradient checkpointing not supported'

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes, global_pool=None):
#         self.num_classes = num_classes
#         if global_pool is not None:
#             assert global_pool in ('token', 'avg')
#             self.global_pool = global_pool
#         self.head = nn.ModuleList(
#             [nn.Linear(self.embed_dim[i], num_classes) if num_classes > 0 else nn.Identity() for i in
#              range(self.num_branches)])

#     def forward_features(self, x) -> List[torch.Tensor]:
#         B = x.shape[0]
#         xs = []
#         for i, patch_embed in enumerate(self.patch_embed):
#             x_ = x
#             ss = self.img_size_scaled[i]
#             x_ = scale_image(x_, ss, self.crop_scale)
#             x_ = patch_embed(x_)
#             cls_tokens = self.cls_token_0 if i == 0 else self.cls_token_1  # hard-coded for torch jit script
#             cls_tokens = cls_tokens.expand(B, -1, -1)
#             x_ = torch.cat((cls_tokens, x_), dim=1)
#             pos_embed = self.pos_embed_0 if i == 0 else self.pos_embed_1  # hard-coded for torch jit script
#             x_ = x_ + pos_embed
#             x_ = self.pos_drop(x_)
#             xs.append(x_)

#         for i, blk in enumerate(self.blocks):
#             xs = blk(xs)

#         # NOTE: was before branch token section, move to here to assure all branch token are before layer norm
#         xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
#         return xs

#     def forward_head(self, xs: List[torch.Tensor], pre_logits: bool = False) -> torch.Tensor:
#         xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
#         if pre_logits or isinstance(self.head[0], nn.Identity):
#             return torch.cat([x for x in xs], dim=1)
#         return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)

#     def forward(self, x):
#         xs = self.forward_features(x)
#         x = self.forward_head(xs)
#         return x




# def _create_crossvit(variant, pretrained=False, **kwargs):
#     default_cfg = default_cfgs[variant]
#     default_num_classes = default_cfg["num_classes"]
#     default_img_size = default_cfg["input_size"][-1]

#     num_classes = kwargs.pop("num_classes", default_num_classes)
#     img_size = kwargs.pop("img_size", default_img_size)
#     repr_size = kwargs.pop("representation_size", None)
#     if repr_size is not None and num_classes != default_num_classes:
#         # Remove representation layer if fine-tuning. This may not always be the desired action,
#         # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
#         _logger.warning("Removing representation layer for fine-tuning.")
#         repr_size = None

#     model_cls = CrossViT
#     model = model_cls(
#         img_size=img_size,
#         num_classes=num_classes,
#         representation_size=repr_size,
#         **kwargs,
#     )
#     model.default_cfg = default_cfg
#     if pretrained:

#         load_pretrained(
#             model,
#             default_cfg,
#             num_classes=num_classes,
#             in_chans=kwargs.get("in_chans", 3),
#             filter_fn=partial(checkpoint_filter_fn, model=model),
#             strict=False,
#         )

#     return model
