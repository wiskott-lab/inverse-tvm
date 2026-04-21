import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers.mlp import Mlp


class InverseViTEncoder(nn.Module):

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_norm=False, init_values=None,
                 proj_drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 mlp_layer=Mlp, depth=12):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                  init_values=init_values, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, mlp_layer=mlp_layer)
            for i in range(depth)])

    def forward(self, x):
        return self.blocks(x)
