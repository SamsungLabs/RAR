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
    
    def forward(self, queries, image_tokens, key_padding_mask=None):
        # self-attention for queries
        q_self, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_self)
        
        # Cross-attention: queries attend to image tokens
        q_across, _ = self.cross_attn(queries, image_tokens, image_tokens, key_padding_mask=key_padding_mask)
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

class TokenProjector(nn.Module):
    def __init__(self, d_in=1024, d_out=768, hidden=2048, dropout=0.0): 
        """ recommend: hidden = 1536 or 2048 """
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, d_out, bias=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # init
        init.constant_(self.ln.weight, 1.0)
        init.constant_(self.ln.bias, 0.0)
        init.trunc_normal_(self.fc1.weight, std=0.02)
        init.trunc_normal_(self.fc2.weight, std=0.02)
    
    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return self.gamma * h

def check_inf(name, value):
    if not torch.isfinite(value).all():
        print("\n===================================================")
        print(f"Error: {name} is not finite. There are NAN existing.")   

def init_transformer_module(module, n_layers):
    for name, m in module.named_modules():
        if isinstance(m, nn.LayerNorm):
            init.constant_(m.weight, 1.0)
            init.constant_(m.bias, 0.0)
            m.eps = 1e-6
        
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)
            
            lname = name.lower()
            if lname.endswith('out_proj') or lname.endswith('fc2') or lname.endswith('proj'):
                m.weight.data.mul_(1.0 / math.sqrt(2.0 * n_layers))

class QFormer(nn.Module):
    """
    Align to Text Encoder in SD3.5:
    clip_l: torch.Size([2, 77, 768]) torch.Size([2, 768])
    clip_g: torch.Size([2, 77, 1280]) torch.Size([2, 1280])
    t5_xxl: torch.Size([2, 77, 4096])
    condition: context: torch.Size([2, 154, 4096]), y: torch.Size([2, 2048])
    """
    def __init__(
        self,
        in_ch = 4096,
        num_input_tokens = 400,
        num_query_tokens = 77,
        hidden_dim = 1024,
        layers = 8,
        heads = 16,
    ):
        # CLIP: 224 14 1024 24 16 768
        # sin positional embedding
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.num_query_tokens = num_query_tokens
        self.in_ch = in_ch 
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Input Projector (align dimension to transformer)
        self.input_norm = LayerNorm(in_ch)
        self.input_proj = nn.Linear(in_ch, hidden_dim)
        init.constant_(self.input_norm.weight, 1.0)
        init.constant_(self.input_norm.bias, 0.0)
        self.input_norm.eps = 1e-6
        init.trunc_normal_(self.input_proj.weight, std=0.02)

        # Query
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim) * 0.02)

        # Cross-Attention
        self.transformer = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, heads) for _ in range(layers)
        ])
        init_transformer_module(self.transformer, n_layers=layers)

        # Output Projector
        self.clip_g_proj = TokenProjector(d_in=hidden_dim, d_out=1280, hidden=2048)
        self.clip_l_proj = TokenProjector(d_in=hidden_dim, d_out=768, hidden=2048)
        self.t5_xxl_proj = TokenProjector(d_in=hidden_dim, d_out=4096, hidden=2048)
        
        # Pooled Projector
        self.clip_g_pool = MultiheadAttnPooling(1280, num_heads=heads, num_queries=1)
        self.clip_g_norm = LayerNorm(1280)
        self.clip_g_line = nn.Parameter(torch.randn(1280, 1280)* 0.02)

        self.clip_l_pool = MultiheadAttnPooling(768, num_heads=heads, num_queries=1)
        self.clip_l_norm = LayerNorm(768)
        self.clip_l_line = nn.Parameter(torch.randn(768, 768)* 0.02)
        
    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        input: torch.Size([1, 400, 4096]) torch.Size([1, 77, 1024])
        kv_token: torch.Size([1, 400, 1024])
        cross_attn: torch.Size([1, 77, 1024])
        project: torch.Size([1, 77, 1280]) torch.Size([1, 77, 768]) torch.Size([1, 77, 4096])
        g_pooled: torch.Size([1, 1, 1280])
        l_pooled: torch.Size([1, 1, 768])
        """
        # project input token to Q_dim
        kv_token = self.input_proj(self.input_norm(x)) # (b, 400, 4096) -> (b, 400, 1024)

        # cross-attention
        batch_size = x.size(0)
        queries = self.query_tokens.repeat(batch_size, 1, 1)  # (b, 77, 1024)
        for layer in self.transformer:
            queries = layer(queries, kv_token, key_padding_mask=key_padding_mask)

        # output project
        g_out = self.clip_g_proj(queries)    # (b, 77, 1280)
        l_out = self.clip_l_proj(queries)    # (b, 77, 768)
        t5_out = self.t5_xxl_proj(queries)   # (b, 77, 4096)

        # pooled project 1     
        g_pooled = self.clip_g_pool(g_out)   # torch.Size([b, 1, 1280])
        g_pooled = self.clip_g_norm(g_pooled)
        g_pooled = g_pooled @ self.clip_g_line   # torch.Size([b, 1280])
        g_pooled = g_pooled[:, 0, :]

        # pooled project 2      
        l_pooled = self.clip_l_pool(l_out)   # torch.Size([b, 1, 768])
        l_pooled = self.clip_l_norm(l_pooled)
        l_pooled = l_pooled @ self.clip_l_line  # torch.Size([b, 768])
        l_pooled = l_pooled[:, 0, :]

        return {
            "g": (g_out, g_pooled),
            "l": (l_out, l_pooled),
            "t5xxl": (t5_out, None)
        }

    def get_cond(self, x, key_padding_mask=None):
        tokens = self(x, key_padding_mask)
        l_out, l_pooled = tokens["l"]
        g_out, g_pooled = tokens["g"]
        t5_out, _ = tokens["t5xxl"]
        lg_out = torch.cat([l_out, g_out], dim=-1)  # (b, 77, 2048)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1])) # (b, 77, 4096)
        context = torch.cat([lg_out, t5_out], dim=-2) # (b, 77+77, 4096)
        y = torch.cat((l_pooled, g_pooled), dim=-1)   # (b, 2048)  
        return context, y   

