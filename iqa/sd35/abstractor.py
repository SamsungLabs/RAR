from collections import OrderedDict
from typing import Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init 

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # orig_type = x.dtype
        ret = super().forward(
            x
        )  # .type(torch.float16))        # Warning: Originally avoid fp32 in clip
        return ret  # .type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
    
    def forward(self, queries, image_tokens):
        # self-attention for queries
        q_self, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_self)
        
        # Cross-attention: queries attend to image tokens
        q_across, _ = self.cross_attn(queries, image_tokens, image_tokens)
        queries = self.norm2(queries + q_across)
        
        # MLP for output
        queries = self.norm3(queries + self.mlp(queries))
        return queries

class MultiheadAttnPooling(nn.Module):
    def __init__(self, dim, num_heads=8, num_queries=1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, num_queries, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, local_tokens):
        """
        local_tokens: [B, N, D]
        return: global_token: [B, 1, D]
        """
        B = local_tokens.size(0)
        query = self.query_token.expand(B, -1, -1)
        global_token, _ = self.attn(query, local_tokens, local_tokens)
        return global_token

def interpolate_embedding(img_size, patch_size, embedding, interpolation_mode="bicubic"):
    h, w = img_size
    assert h % patch_size == 0 and w % patch_size == 0
    h_grid = h // patch_size
    w_grid = w // patch_size
    embedding_token = embedding[:1, :]
    embedding_img = embedding[1:, :]
    n, c  = embedding_img.shape
    n_1d = int(math.sqrt(n))
    assert n_1d * n_1d == n
    embedding_img = embedding_img.permute(1, 0).reshape(1, c, n_1d, n_1d)
    embedding_img_new = F.interpolate(
        embedding_img,
        size=(h_grid, w_grid),
        mode=interpolation_mode,
        align_corners=True,
    ).squeeze(0)
    embedding_img_new = embedding_img_new.reshape(c, -1)
    embedding_img_new = embedding_img_new.permute(1, 0)
    embedding_new = torch.cat([embedding_token, embedding_img_new], dim=0)
    return embedding_new

class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)

class SimpleAbstractorModel(nn.Module):
    def __init__(
        self,
        in_ch = 16,
        patch_size = 1,
        width = 1024,
        layers = 8,
        heads = 16,
        output_dim = 768,
        pos = True,
    ):
        # CLIP: input:(3, 224, 224) -> local:(1024, 16, 16) -> global:(768, 1)
        # AutoencoderDC: input:(3, 512, 512) -> encode:(32, 16, 16) -> conv: (1024, 16, 16) -> local: (16*16, 1024) -> global: (1, 768)
        # sin positional embedding
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size)

        scale = width**-0.5
        self.local_norm = LayerNorm(width)
        self.local_proj = nn.Parameter(scale * torch.randn(width, width))

        self.weight_proj = nn.Linear(width, 1)
        self.global_norm = LayerNorm(width)
        self.global_proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.pos = pos
        
    def forward(self, x: torch.Tensor):
        # project to token
        x = self.conv1(x)  # shape = [*, width, grid, grid], grid=1/16 # 2: torch.Size([1, 1024, p, p])
        B, _, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  # BDL
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 3: torch.Size([1, p*p, 1024])  # BLD

        # positional embedding
        if self.pos:
            pos_emb = PositionEmbeddingSine((H, W), self.width // 2, normalize=True) # (H X W) X C
            pos_emb = pos_emb(x)[None, :, :].repeat(B, 1, 1).to(dtype=x.dtype, device=x.device) # BLD
            x = x + pos_emb    # torch.Size([1, 25, 1024]) torch.float16

        # local
        x = self.local_norm(x) # local: BLD # 6: torch.Size([1, p*p, 1024]) 
        if self.local_proj is not None: 
            x = x @ self.local_proj         # 7: torch.Size([1, p*p, 1024]) 
            
        # global
        score = self.weight_proj(x)  # torch.Size([1, p*p, 1])
        weights = F.softmax(score.squeeze(-1), dim=-1)  # torch.Size([1, p*p])
        x = torch.bmm(weights.unsqueeze(1), x) # torch.Size([1, 1, 1024])
        x = self.global_norm(x)        # local: BLD # 6: torch.Size([1, 1, 1024])
        if self.global_proj is not None:
            x = x @ self.global_proj   # 7: torch.Size([1, 1, 768])
        return x                       # torch.Size([1, 1, 768])

    def forward_patch_features(self, x: torch.Tensor):
        # project to token
        x = self.conv1(x)  # shape = [*, width, grid, grid], grid=1/16 # 2: torch.Size([1, 1024, p, p])
        B, _, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  # BDL
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 3: torch.Size([1, p*p, 1024])  # BLD

        # positional embedding
        if self.pos:
            pos_emb = PositionEmbeddingSine((H, W), self.width // 2, normalize=True) # (H X W) X C
            pos_emb = pos_emb(x)[None, :, :].repeat(B, 1, 1).to(dtype=x.dtype, device=x.device) # BLD
            x = x + pos_emb    # torch.Size([1, 25, 1024]) torch.float16

        # local
        x = self.local_norm(x) # local: BLD # torch.Size([1, p*p, 1024]) 
        if self.local_proj is not None: 
            x = x @ self.local_proj         # torch.Size([1, p*p, 1024]) 
        return x
    
    def forward_feats(self, x: torch.Tensor):
        # input: torch.Size([1, 8, 80/8, 80/8]) torch.float16
        feats = []
        # project to token
        x1 = self.conv1(x)  # shape = [*, width, grid, grid], grid=1/16 # 2: torch.Size([1, 1024, p, p])
        B, _, H, W = x1.shape
        x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)  # shape = [*, width, grid ** 2]  # BDL
        x1 = x1.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 3: torch.Size([1, p*p, 1024])  # BLD
        feats.append(x1)  # torch.Size([B, p*p, 1024])

        # positional embedding
        if self.pos:
            pos_emb = PositionEmbeddingSine((H, W), self.width // 2, normalize=True) # (H X W) X C
            pos_emb = pos_emb(x1)[None, :, :].repeat(B, 1, 1).to(dtype=x1.dtype, device=x1.device) # BLD
            x1 = x1 + pos_emb    # torch.Size([1, 25, 1024]) torch.float16

        # local
        x2 = self.local_norm(x1) # local: BLD # 6: torch.Size([1, p*p, 1024]) 
        if self.local_proj is not None: 
            x2 = x2 @ self.local_proj         # 7: torch.Size([1, p*p, 1024]) 
        feats.append(x2)  # torch.Size([B, p*p, 1024])
        
        # global
        score = self.weight_proj(x2)  # torch.Size([1, p*p, 1])
        weights = F.softmax(score.squeeze(-1), dim=-1)  # torch.Size([1, p*p])
        x3 = torch.bmm(weights.unsqueeze(1), x2) # torch.Size([1, 1, 1024])
        x3 = self.global_norm(x3)        # local: BLD # 6: torch.Size([1, 1, 1024])
        if self.global_proj is not None:
            x3 = x3 @ self.global_proj   # 7: torch.Size([1, 1, 768])
        feats.append(x3)  # torch.Size([B, 1, 768])
        return feats  # [Patch: (B, p*p, Dim), Local: (B, p*p, Dim), Global: (B, 1, 768)]  

def check_inf(name, value):
    if not torch.isfinite(value).all():
        print("\n===================================================")
        print(f"Error: {name} is not finite. There are NAN existing.")   

class FormerAbstractorModel(nn.Module):
    def __init__(
        self,
        in_ch = 16,
        in_patch = 8,
        patch_size = 16,
        num_query_tokens = 1,
        width = 1024,
        layers = 8,
        qlayers = 1,
        heads = 16,
        output_dim = 768,
    ):
        # CLIP: 224 14 1024 24 16 768
        # sin positional embedding
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.patch_size = patch_size 
        self.in_patch = in_patch
        self.patch_proj = int(patch_size // in_patch) # 32 // 32 = 1
        self.output_dim = output_dim
        self.width = width
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=width,
            kernel_size=patch_size // in_patch,
            stride=patch_size // in_patch,
            bias=False,
        )
        scale = width**-0.5
        init.constant_(self.conv1.weight, 1/in_ch)

        # for local
        self.ln_pre = LayerNorm(width)
        self.local_attn = Transformer(width, layers, heads)
        
        # for global
        self.attn_pool = MultiheadAttnPooling(width, num_heads=heads, num_queries=1)
        self.global_norm = LayerNorm(width)
        self.global_proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
    def forward(self, x: torch.Tensor):
        # project to token
        x = self.conv1(x)  # shape = [*, width, grid, grid], grid=1/16 # 2: torch.Size([1, 1024, p, p])
        B, _, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  # BDL
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 3: torch.Size([1, p*p, 1024])  # BLD        
        
        pos_emb = PositionEmbeddingSine((H, W), self.width // 2, normalize=True) # (H X W) X C
        pos_emb = pos_emb(x)[None, :, :].repeat(B, 1, 1).to(dtype=x.dtype, device=x.device) # BLD
        x = x + pos_emb    # torch.Size([1, 25, 1024]) torch.float16
    
        # Local Attention
        x = self.ln_pre(x) # 4: torch.Size([1, p*p, 1024])
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.local_attn(x)
        x = x.permute(1, 0, 2)  # LND -> NLD #  5: torch.Size([1, p*p, 1024])
        
        # Global Attention
        x = self.attn_pool(x)   # torch.Size([1, 1, 1024])
        x = self.global_norm(x)
        x = x @ self.global_proj   # 7: torch.Size([1, 768])
        return x

    def forward_patch_features(self, x: torch.Tensor):
        # project to token
        x = self.conv1(x)  # shape = [*, width, grid, grid], grid=1/16 # 2: torch.Size([1, 1024, p, p])
        B, _, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]  # BDL
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 3: torch.Size([1, p*p, 1024])  # BLD        

        pos_emb = PositionEmbeddingSine((H, W), self.width // 2, normalize=True) # (H X W) X C
        pos_emb = pos_emb(x)[None, :, :].repeat(B, 1, 1).to(dtype=x.dtype, device=x.device) # BLD
        x = x + pos_emb    # torch.Size([1, 25, 1024]) torch.float16
    
        # Local Attention
        x = self.ln_pre(x) # 4: torch.Size([1, p*p, 1024])
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.local_attn(x)
        x = x.permute(1, 0, 2)  # LND -> NLD #  5: torch.Size([1, p*p, 1024])
        return x
    
    def forward_feats(self, x: torch.Tensor):
        feats = []

        # project to token
        x1 = self.conv1(x)  # shape = [*, width, grid, grid], grid=1/16 # 2: torch.Size([1, 1024, p, p])
        B, _, H, W = x1.shape
        x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)  # shape = [*, width, grid ** 2]  # BDL
        x1 = x1.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  # 3: torch.Size([1, p*p, 1024])  # BLD        
        feats.append(x1)

        pos_emb = PositionEmbeddingSine((H, W), self.width // 2, normalize=True) # (H X W) X C
        pos_emb = pos_emb(x1)[None, :, :].repeat(B, 1, 1).to(dtype=x1.dtype, device=x1.device) # BLD
        x1 = x1 + pos_emb    # torch.Size([1, 25, 1024]) torch.float16
    
        # Local Attention
        x2 = self.ln_pre(x1) # 4: torch.Size([1, p*p, 1024])
        x2 = x2.permute(1, 0, 2)  # NLD -> LND
        x2 = self.local_attn(x2)
        x2 = x2.permute(1, 0, 2)  # LND -> NLD #  5: torch.Size([1, p*p, 1024])
        feats.append(x2)

        # Global Attention
        x3 = self.attn_pool(x2)   # torch.Size([1, 1, 1024])
        x3 = self.global_norm(x3)
        x3 = x3 @ self.global_proj   # 7: torch.Size([1, 768])
        feats.append(x3)

        return feats # [Patch: (B, p*p, Dim), Local: (B, p*p, Dim), Global: (B, 1, 768)] 
