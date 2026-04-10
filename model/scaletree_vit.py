"""Efficient Vision Transformer variants built with ScaleTree FFN blocks."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

try:
    from scaring_laws.module import ScaleTreeFFN, ScaleTreeFFNConfig
except ImportError:
    from scaring_laws.models import ScaleTreeFFN, ScaleTreeFFNConfig


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor.div(keep_prob)


class PatchEmbed(nn.Module):
    """Patch embedding via a single strided convolution."""

    def __init__(
        self,
        *,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        image_size = _to_2tuple(image_size)
        patch_size = _to_2tuple(patch_size)
        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError(
                "image_size must be divisible by patch_size. "
                f"Got image_size={image_size}, patch_size={patch_size}."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(
                "PatchEmbed expects input shaped [batch, channels, height, width]. "
                f"Got shape={tuple(x.shape)}."
            )
        x = self.proj(x)
        grid_size = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size


class EfficientSelfAttention(nn.Module):
    """Multi-head self-attention with Flash Attention aware backend selection."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        attention_backend: Literal["auto", "flash_attn", "torch_flash", "sdpa"] = "auto",
        flash_deterministic: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads. "
                f"Got embed_dim={embed_dim}, num_heads={num_heads}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_backend = attention_backend
        self.flash_deterministic = flash_deterministic

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        attn_output = self._run_attention(qkv)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        return self.proj_dropout(self.proj(attn_output))

    def _run_attention(self, qkv: Tensor) -> Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        if self._should_use_external_flash_attention(qkv):
            return flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=self.scale,
                causal=False,
                deterministic=self.flash_deterministic,
            )

        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self._should_use_torch_flash_attention(query):
            try:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    attn_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        scale=self.scale,
                    )
                return attn_output.transpose(1, 2)
            except RuntimeError:
                if self.attention_backend == "torch_flash":
                    raise

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            scale=self.scale,
        )
        return attn_output.transpose(1, 2)

    def _should_use_external_flash_attention(self, qkv: Tensor) -> bool:
        if self.attention_backend not in {"auto", "flash_attn"}:
            return False
        if flash_attn_qkvpacked_func is None:
            return False
        return qkv.is_cuda and qkv.dtype in {torch.float16, torch.bfloat16}

    def _should_use_torch_flash_attention(self, query: Tensor) -> bool:
        if self.attention_backend not in {"auto", "torch_flash"}:
            return False
        return query.is_cuda and query.dtype in {torch.float16, torch.bfloat16}


class ScaleTreeViTBlock(nn.Module):
    """Pre-norm ViT block with ScaleTree FFN replacing the dense MLP."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        tree_ffn_config: ScaleTreeFFNConfig,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        attention_backend: Literal["auto", "flash_attn", "torch_flash", "sdpa"] = "auto",
        flash_deterministic: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = EfficientSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            attention_backend=attention_backend,
            flash_deterministic=flash_deterministic,
        )
        self.ffn = ScaleTreeFFN(tree_ffn_config)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        if init_values is not None:
            if init_values <= 0.0:
                raise ValueError("init_values must be positive when specified.")
            self.gamma1 = nn.Parameter(torch.full((embed_dim,), init_values))
            self.gamma2 = nn.Parameter(torch.full((embed_dim,), init_values))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(
        self,
        x: Tensor,
        *,
        route_mode: str | None = None,
        temperature: float | None = None,
        top_k_leaves: int | None = None,
    ) -> Tensor:
        attn_out = self.attn(self.norm1(x))
        if self.gamma1 is not None:
            attn_out = attn_out * self.gamma1
        x = x + self.drop_path1(attn_out)

        ffn_out = self.ffn(
            self.norm2(x),
            route_mode=route_mode,
            temperature=temperature,
            top_k_leaves=top_k_leaves,
        )
        if self.gamma2 is not None:
            ffn_out = ffn_out * self.gamma2
        x = x + self.drop_path2(ffn_out)
        return x


@dataclass(frozen=True, slots=True)
class ScaleTreeViTConfig:
    """Configuration for an efficient ViT with ScaleTree FFN blocks."""

    image_size: int | tuple[int, int] = 224
    patch_size: int | tuple[int, int] = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    attention_backend: Literal["auto", "flash_attn", "torch_flash", "sdpa"] = "auto"
    flash_deterministic: bool = False
    init_values: float | None = None
    global_pool: Literal["cls", "mean"] = "cls"
    use_cls_token: bool = True
    pos_embed_dropout: float = 0.0
    patch_embed_bias: bool = True
    ffn_num_groups: int = 4
    ffn_routing_dim: int = 16
    ffn_num_stages: int = 2
    ffn_trees_per_group: int = 2
    ffn_tree_depth: int = 3
    ffn_tree_width: int | None = None
    ffn_micro_block_size: int = 4
    ffn_router_temperature: float = 1.0
    ffn_training_route_mode: Literal["soft", "straight_through"] = "soft"
    ffn_inference_route_mode: Literal["soft", "hard"] = "hard"
    ffn_top_k_leaves: int | None = None
    ffn_summary_bias: bool = True
    ffn_leaf_bias: bool = True
    ffn_output_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.global_pool not in {"cls", "mean"}:
            raise ValueError("global_pool must be 'cls' or 'mean'.")
        if not self.use_cls_token and self.global_pool == "cls":
            raise ValueError("global_pool='cls' requires use_cls_token=True.")
        if self.attention_backend not in {"auto", "flash_attn", "torch_flash", "sdpa"}:
            raise ValueError(
                "attention_backend must be one of 'auto', 'flash_attn', "
                "'torch_flash', or 'sdpa'."
            )

    def build_tree_ffn_config(self) -> ScaleTreeFFNConfig:
        return ScaleTreeFFNConfig(
            embed_dim=self.embed_dim,
            num_groups=self.ffn_num_groups,
            routing_dim=self.ffn_routing_dim,
            num_stages=self.ffn_num_stages,
            trees_per_group=self.ffn_trees_per_group,
            tree_depth=self.ffn_tree_depth,
            tree_width=self.ffn_tree_width,
            micro_block_size=self.ffn_micro_block_size,
            router_temperature=self.ffn_router_temperature,
            training_route_mode=self.ffn_training_route_mode,
            inference_route_mode=self.ffn_inference_route_mode,
            top_k_leaves=self.ffn_top_k_leaves,
            summary_bias=self.ffn_summary_bias,
            leaf_bias=self.ffn_leaf_bias,
            output_dropout=self.ffn_output_dropout,
        )


class ScaleTreeViT(nn.Module):
    """Vision Transformer with ScaleTree FFN blocks."""

    def __init__(self, config: ScaleTreeViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            bias=config.patch_embed_bias,
        )
        self.num_prefix_tokens = 1 if config.use_cls_token else 0
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            if config.use_cls_token
            else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.patch_embed.num_patches + self.num_prefix_tokens,
                config.embed_dim,
            )
        )
        self.pos_drop = nn.Dropout(config.pos_embed_dropout)
        self.patch_drop = nn.Dropout(config.drop_rate)

        drop_path_rates = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        tree_ffn_config = config.build_tree_ffn_config()
        self.blocks = nn.ModuleList(
            [
                ScaleTreeViTBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    tree_ffn_config=tree_ffn_config,
                    qkv_bias=config.qkv_bias,
                    attn_dropout=config.attn_drop_rate,
                    proj_dropout=config.drop_rate,
                    drop_path=drop_path_rates[index],
                    init_values=config.init_values,
                    attention_backend=config.attention_backend,
                    flash_deterministic=config.flash_deterministic,
                )
                for index in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = (
            nn.Linear(config.embed_dim, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(
        self,
        x: Tensor,
        *,
        route_mode: str | None = None,
        temperature: float | None = None,
        top_k_leaves: int | None = None,
    ) -> Tensor:
        x, grid_size = self.patch_embed(x)
        batch_size = x.shape[0]

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self._interpolate_pos_embed(grid_size, x.dtype)
        x = self.patch_drop(self.pos_drop(x))

        for block in self.blocks:
            x = block(
                x,
                route_mode=route_mode,
                temperature=temperature,
                top_k_leaves=top_k_leaves,
            )

        x = self.norm(x)
        if self.config.global_pool == "mean":
            patch_tokens = x[:, self.num_prefix_tokens :, :]
            return patch_tokens.mean(dim=1)
        if self.cls_token is None:
            raise RuntimeError("CLS pooling requested but the model has no CLS token.")
        return x[:, 0]

    def forward(
        self,
        x: Tensor,
        *,
        route_mode: str | None = None,
        temperature: float | None = None,
        top_k_leaves: int | None = None,
    ) -> Tensor:
        features = self.forward_features(
            x,
            route_mode=route_mode,
            temperature=temperature,
            top_k_leaves=top_k_leaves,
        )
        return self.head(features)

    def _interpolate_pos_embed(self, grid_size: tuple[int, int], dtype: torch.dtype) -> Tensor:
        if grid_size == self.patch_embed.grid_size:
            return self.pos_embed.to(dtype=dtype)

        prefix = self.pos_embed[:, : self.num_prefix_tokens]
        patch_pos = self.pos_embed[:, self.num_prefix_tokens :]
        patch_pos = patch_pos.reshape(
            1,
            self.patch_embed.grid_size[0],
            self.patch_embed.grid_size[1],
            self.config.embed_dim,
        ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=grid_size,
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1,
            grid_size[0] * grid_size[1],
            self.config.embed_dim,
        )
        return torch.cat((prefix, patch_pos), dim=1).to(dtype=dtype)

    def no_weight_decay(self) -> set[str]:
        names = {"pos_embed"}
        if self.cls_token is not None:
            names.add("cls_token")
        return names


def scaletree_vit_tiny_patch16(**kwargs: object) -> ScaleTreeViT:
    config = ScaleTreeViTConfig(
        embed_dim=192,
        depth=12,
        num_heads=3,
        ffn_num_groups=4,
        ffn_routing_dim=12,
        ffn_num_stages=2,
        ffn_trees_per_group=2,
        ffn_tree_width=8,
        ffn_micro_block_size=4,
        **kwargs,
    )
    return ScaleTreeViT(config)


def scaletree_vit_small_patch16(**kwargs: object) -> ScaleTreeViT:
    config = ScaleTreeViTConfig(
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_num_groups=8,
        ffn_routing_dim=12,
        ffn_num_stages=2,
        ffn_trees_per_group=2,
        ffn_tree_width=8,
        ffn_micro_block_size=4,
        **kwargs,
    )
    return ScaleTreeViT(config)


def scaletree_vit_base_patch16(**kwargs: object) -> ScaleTreeViT:
    config = ScaleTreeViTConfig(
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_num_groups=8,
        ffn_routing_dim=24,
        ffn_num_stages=2,
        ffn_trees_per_group=2,
        ffn_tree_width=8,
        ffn_micro_block_size=8,
        **kwargs,
    )
    return ScaleTreeViT(config)
