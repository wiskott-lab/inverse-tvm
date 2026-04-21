import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from torch.ao.nn.quantized.functional import upsample

from modules.detr.util.misc import NestedTensor
from tools import vit_utils as vu
import json
import torch
import config
from modules.detr.datasets import get_coco_api_from_dataset
from modules.detr.models.detr import SetCriterion
from modules.detr.models.matcher import HungarianMatcher
from modules.detr.datasets.coco_eval import CocoEvaluator
from modules.detr.models.detr import PostProcess
import tools.coco_utils as cu
from modules.detr.datasets import transforms as coco_transforms
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
from itertools import repeat
import collections.abc
from timm.models.swin_transformer import SwinTransformerBlock, PatchEmbed
from typing import Any, Dict, Callable, List, Optional, Set, Tuple, Union


class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv_1 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU()

        self.up_conv_2 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.bn_2 = nn.BatchNorm2d(3)
        self.sig_2 = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.relu_1(self.bn_1(self.up_conv_1(x)))
        x = self.sig_2(self.bn_2(self.up_conv_2(x)))
        return x


class PatchSplitting(nn.Module):
    """Inverse of PatchMerging — upsample features by splitting patches."""

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.LayerNorm,
    ):
        """
        Args:
            dim: Number of input channels (after merging).
            out_dim: Number of output channels (before merging, typically dim // 2).
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim // 2
        self.norm = norm_layer(dim)
        self.expansion = nn.Linear(dim, 4 * self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # normalize and expand channels
        x = self.norm(x)
        x = self.expansion(x)  # (B, H, W, 4*out_dim)

        # rearrange: reverse of the 2x2 merging
        x = x.view(B, H, W, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2, W * 2, self.out_dim)
        return x


class InverseSwinTransformerStage(nn.Module):

    def __init__(
            self, dim: int, out_dim: int, input_resolution: Tuple[int, int],
            output_resolution,
            depth: int,
            upsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size=(7, 7),
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.LayerNorm):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.depth = depth
        self.grad_checkpointing = False
        shift_size = tuple([w // 2 for w in window_size])


        if upsample:
            self.upsample = PatchSplitting(dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock( dim=dim, input_resolution=self.input_resolution, num_heads=num_heads,
                                  head_dim=head_dim,window_size=window_size,
                                  shift_size=0 if (i % 2 == 1) else shift_size,  always_partition=always_partition,
                                  dynamic_mask=dynamic_mask, mlp_ratio=mlp_ratio,  qkv_bias=qkv_bias,
                                  proj_drop=proj_drop, attn_drop=attn_drop,
                                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer) for i in range(depth)])

    def set_input_size(
            self,
            feat_size: Tuple[int, int],
            window_size: int,
            always_partition: Optional[bool] = None,
    ):
        """ Updates the resolution, window size and so the pair-wise relative positions.

        Args:
            feat_size: New input (feature) resolution
            window_size: New window size
            always_partition: Always partition / shift the window
        """
        self.input_resolution = feat_size
        if isinstance(self.downsample, nn.Identity):
            self.output_resolution = feat_size
        else:
            self.output_resolution = tuple(i // 2 for i in feat_size)
        for block in self.blocks:
            block.set_input_size(
                feat_size=self.output_resolution,
                window_size=window_size,
                always_partition=always_partition,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.upsample(x)
        return x


if __name__ == '__main__':
    bb = SwinBackbone()
    a = torch.rand(size=(10, 56, 56, 128))
    b = bb(a)
