"""Attentive pooling / classification heads for online probing.

Faithful, self-contained port of the I-JEPA / V-JEPA ``AttentivePooler`` and
``AttentiveClassifier`` (Meta Platforms, Apache-2.0). The standard transformer
block and MLP are reused from ``timm`` so only the cross-attention pieces are
implemented here.
"""

import math

import torch
import torch.nn as nn
from timm.layers import Mlp, trunc_normal_
from timm.models.vision_transformer import Block


class CrossAttention(nn.Module):
    """Cross-attention: learnable queries attend over the token sequence ``x``."""

    def __init__(self, dim, num_heads=12, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, x):
        B, n, C = q.shape
        _, N, _ = x.shape
        q = (
            self.q(q)
            .reshape(B, n, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        q = (attn @ v).transpose(1, 2).reshape(B, n, C)
        return self.proj_drop(self.proj(q))


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block with a feed-forward MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=dropout
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=dropout,
        )

    def forward(self, q, x):
        q = q + self.xattn(q, self.norm1(x))
        q = q + self.mlp(self.norm2(q))
        return q


class AttentivePooler(nn.Module):
    """Attentive pooler: pools a token sequence into ``num_queries`` vectors."""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True,
        dropout=0.0,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                dropout=dropout,
            )
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout=dropout,
            )

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        norm_layer=norm_layer,
                        proj_drop=dropout,
                        attn_drop=dropout,
                    )
                    for _ in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q


class AttentiveClassifier(nn.Module):
    """Attentive pooler followed by a linear classification head."""

    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        return self.linear(x)
