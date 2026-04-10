"""Theory-guided tree residual feed-forward modules.

This module implements a reusable replacement for Transformer FFN/MLP layers:

    LN(x) -> grouped summary projection -> differentiable oblivious trees
           -> block-diagonal leaf experts -> residual branch output

The implementation is intentionally training-framework agnostic. It exposes a
standalone ``ScaleTreeFFN`` that can replace an FFN/MLP module, plus a small
``ScaleTreeTransformerBlock`` wrapper for pre-norm residual blocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn

RouteMode = Literal["soft", "hard", "straight_through"]


def _build_leaf_bit_table(tree_depth: int) -> Tensor:
    """Return leaf-to-path branch indicators for an oblivious binary tree."""
    num_leaves = 1 << tree_depth
    leaf_ids = torch.arange(num_leaves, dtype=torch.long)
    shifts = torch.arange(tree_depth - 1, -1, -1, dtype=torch.long)
    bits = ((leaf_ids[:, None] >> shifts[None, :]) & 1).to(dtype=torch.float32)
    return bits


@dataclass(slots=True)
class RoutingState:
    """Routing outputs for dense or sparse expert evaluation."""

    leaf_probs: Tensor | None = None
    selected_indices: Tensor | None = None
    selected_probs: Tensor | None = None


@dataclass(frozen=True, slots=True)
class ScaleTreeFFNConfig:
    """Configuration for a grouped multi-stage tree FFN.

    ``tree_width`` is the explicit width of each oblivious tree, measured as
    the number of leaves. When set, it overrides the width implied by
    ``tree_depth`` and the effective depth is inferred as ``log2(tree_width)``.
    """

    embed_dim: int
    num_groups: int
    routing_dim: int
    num_stages: int = 2
    trees_per_group: int = 2
    tree_depth: int = 3
    tree_width: int | None = None
    micro_block_size: int = 4
    init_stage_scale: float = 1.0
    router_temperature: float = 1.0
    training_route_mode: RouteMode = "soft"
    inference_route_mode: Literal["soft", "hard"] = "hard"
    top_k_leaves: int | None = None
    summary_bias: bool = True
    leaf_bias: bool = True
    output_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.num_groups <= 0:
            raise ValueError("num_groups must be positive.")
        if self.embed_dim % self.num_groups != 0:
            raise ValueError(
                "embed_dim must be divisible by num_groups. "
                f"Got embed_dim={self.embed_dim}, num_groups={self.num_groups}."
            )
        if self.routing_dim <= 0:
            raise ValueError("routing_dim must be positive.")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive.")
        if self.trees_per_group <= 0:
            raise ValueError("trees_per_group must be positive.")
        if self.tree_depth <= 0:
            raise ValueError("tree_depth must be positive.")
        if self.tree_width is not None:
            if self.tree_width <= 0:
                raise ValueError("tree_width must be positive when specified.")
            if self.tree_width < 2:
                raise ValueError("tree_width must be at least 2 when specified.")
            if self.tree_width & (self.tree_width - 1):
                raise ValueError(
                    "tree_width must be a power of two for a binary oblivious tree. "
                    f"Got tree_width={self.tree_width}."
                )
        if self.micro_block_size <= 0:
            raise ValueError("micro_block_size must be positive.")
        if self.group_dim % self.micro_block_size != 0:
            raise ValueError(
                "group_dim must be divisible by micro_block_size. "
                f"Got group_dim={self.group_dim}, micro_block_size={self.micro_block_size}."
            )
        if self.router_temperature <= 0.0:
            raise ValueError("router_temperature must be positive.")
        if not 0.0 <= self.output_dropout < 1.0:
            raise ValueError("output_dropout must be in [0, 1).")
        if self.top_k_leaves is not None:
            if self.top_k_leaves <= 0:
                raise ValueError("top_k_leaves must be positive when specified.")
            if self.top_k_leaves > self.num_leaves:
                raise ValueError(
                    "top_k_leaves cannot exceed the total number of leaves. "
                    f"Got top_k_leaves={self.top_k_leaves}, num_leaves={self.num_leaves}."
                )

    @property
    def group_dim(self) -> int:
        return self.embed_dim // self.num_groups

    @property
    def num_leaves(self) -> int:
        return self.tree_width if self.tree_width is not None else 1 << self.tree_depth

    @property
    def effective_tree_depth(self) -> int:
        if self.tree_width is not None:
            return int(math.log2(self.tree_width))
        return self.tree_depth

    @property
    def num_micro_blocks(self) -> int:
        return self.group_dim // self.micro_block_size


class ObliviousTreeRouter(nn.Module):
    """Differentiable binary router for a bank of oblivious trees."""

    def __init__(self, config: ScaleTreeFFNConfig) -> None:
        super().__init__()
        self.config = config
        self.split_weight = nn.Parameter(
            torch.zeros(
                config.num_groups,
                config.trees_per_group,
                config.effective_tree_depth,
                config.routing_dim,
            )
        )
        self.split_bias = nn.Parameter(
            torch.zeros(
                config.num_groups,
                config.trees_per_group,
                config.effective_tree_depth,
            )
        )
        self.register_buffer(
            "leaf_bits",
            _build_leaf_bit_table(config.effective_tree_depth),
            persistent=False,
        )
        self.register_buffer(
            "bit_shifts",
            torch.arange(
                config.effective_tree_depth - 1,
                -1,
                -1,
                dtype=torch.long,
            ),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.split_weight)
        nn.init.zeros_(self.split_bias)

    def forward(
        self,
        summary: Tensor,
        *,
        route_mode: RouteMode,
        temperature: float,
        top_k_leaves: int | None,
    ) -> RoutingState:
        if route_mode not in {"soft", "hard", "straight_through"}:
            raise ValueError(f"Unsupported route_mode: {route_mode}")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        logits = torch.einsum("tgr,gmdr->tgmd", summary, self.split_weight)
        logits = logits + self.split_bias
        decision_probs = torch.sigmoid(logits / temperature)

        if route_mode == "hard":
            hard = decision_probs >= 0.5
            selected_indices = self._decode_leaf_indices(hard).unsqueeze(-1)
            selected_probs = torch.ones_like(
                selected_indices,
                dtype=summary.dtype,
                device=summary.device,
            )
            return RoutingState(
                selected_indices=selected_indices,
                selected_probs=selected_probs,
            )
        elif route_mode == "straight_through":
            hard = (decision_probs >= 0.5).to(dtype=decision_probs.dtype)
            routed = hard - decision_probs.detach() + decision_probs
        else:
            routed = decision_probs

        leaf_bits = self.leaf_bits.to(device=summary.device, dtype=torch.bool)
        branch_probs = torch.where(
            leaf_bits.view(
                1,
                1,
                1,
                self.config.num_leaves,
                self.config.effective_tree_depth,
            ),
            routed.unsqueeze(-2),
            1.0 - routed.unsqueeze(-2),
        )
        leaf_probs = branch_probs.prod(dim=-1)

        if top_k_leaves is not None and top_k_leaves < self.config.num_leaves:
            top_probs, top_indices = leaf_probs.topk(top_k_leaves, dim=-1)
            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            return RoutingState(
                selected_indices=top_indices,
                selected_probs=top_probs,
            )

        return RoutingState(leaf_probs=leaf_probs)

    def _decode_leaf_indices(self, hard_decisions: Tensor) -> Tensor:
        bit_values = hard_decisions.to(dtype=torch.long)
        powers = (1 << self.bit_shifts).view(1, 1, 1, -1)
        return (bit_values * powers).sum(dim=-1)


class BlockDiagonalExpertBank(nn.Module):
    """Leaf experts with block-diagonal micro-mixers."""

    def __init__(self, config: ScaleTreeFFNConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.zeros(
                config.num_groups,
                config.trees_per_group,
                config.num_leaves,
                config.num_micro_blocks,
                config.micro_block_size,
                config.micro_block_size,
            )
        )
        if config.leaf_bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    config.num_groups,
                    config.trees_per_group,
                    config.num_leaves,
                    config.num_micro_blocks,
                    config.micro_block_size,
                )
            )
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, grouped_input: Tensor, routing: RoutingState) -> Tensor:
        num_tokens = grouped_input.shape[0]
        block_input = grouped_input.reshape(
            num_tokens,
            self.config.num_groups,
            self.config.num_micro_blocks,
            self.config.micro_block_size,
        )

        if routing.leaf_probs is not None:
            mixed_output = torch.einsum(
                "tgmk,gmkqij,tgqj->tgmqi",
                routing.leaf_probs,
                self.weight,
                block_input,
            )
            if self.bias is not None:
                mixed_output = mixed_output + torch.einsum(
                    "tgmk,gmkqi->tgmqi",
                    routing.leaf_probs,
                    self.bias,
                )
        else:
            if routing.selected_indices is None or routing.selected_probs is None:
                raise ValueError("Sparse routing requires selected indices and probabilities.")
            mixed_output = self._forward_sparse(
                block_input,
                routing.selected_indices,
                routing.selected_probs,
            )

        group_output = mixed_output.mean(dim=2)
        return group_output.reshape(
            num_tokens,
            self.config.num_groups,
            self.config.group_dim,
        )

    def _forward_sparse(
        self,
        block_input: Tensor,
        selected_indices: Tensor,
        selected_probs: Tensor,
    ) -> Tensor:
        num_tokens = block_input.shape[0]
        num_selected = selected_indices.shape[-1]
        num_group_trees = self.config.num_groups * self.config.trees_per_group

        flat_group_tree = (
            torch.arange(num_group_trees, device=block_input.device)
            .view(1, self.config.num_groups, self.config.trees_per_group, 1)
            .expand(num_tokens, -1, -1, num_selected)
            .reshape(-1)
        )
        flat_selected = selected_indices.reshape(-1)

        flat_weight = self.weight.reshape(
            num_group_trees,
            self.config.num_leaves,
            self.config.num_micro_blocks,
            self.config.micro_block_size,
            self.config.micro_block_size,
        )
        gathered_weight = flat_weight[flat_group_tree, flat_selected].reshape(
            num_tokens,
            self.config.num_groups,
            self.config.trees_per_group,
            num_selected,
            self.config.num_micro_blocks,
            self.config.micro_block_size,
            self.config.micro_block_size,
        )

        sparse_output = torch.einsum(
            "tgmsqij,tgqj->tgmsqi",
            gathered_weight,
            block_input,
        )

        if self.bias is not None:
            flat_bias = self.bias.reshape(
                num_group_trees,
                self.config.num_leaves,
                self.config.num_micro_blocks,
                self.config.micro_block_size,
            )
            gathered_bias = flat_bias[flat_group_tree, flat_selected].reshape(
                num_tokens,
                self.config.num_groups,
                self.config.trees_per_group,
                num_selected,
                self.config.num_micro_blocks,
                self.config.micro_block_size,
            )
            sparse_output = sparse_output + gathered_bias

        weighted_output = sparse_output * selected_probs.unsqueeze(-1).unsqueeze(-1)
        return weighted_output.sum(dim=3)


class ScaleTreeStage(nn.Module):
    """One stage of grouped routing plus block-diagonal experts."""

    def __init__(self, config: ScaleTreeFFNConfig) -> None:
        super().__init__()
        self.config = config
        self.summary_weight = nn.Parameter(
            torch.empty(
                config.num_groups,
                config.routing_dim,
                config.group_dim,
            )
        )
        if config.summary_bias:
            self.summary_bias = nn.Parameter(
                torch.zeros(config.num_groups, config.routing_dim)
            )
        else:
            self.register_parameter("summary_bias", None)

        self.router = ObliviousTreeRouter(config)
        self.experts = BlockDiagonalExpertBank(config)
        self.stage_scale = nn.Parameter(torch.full((), config.init_stage_scale))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.summary_weight)
        if self.summary_bias is not None:
            nn.init.zeros_(self.summary_bias)
        self.router.reset_parameters()
        self.experts.reset_parameters()
        with torch.no_grad():
            self.stage_scale.fill_(self.config.init_stage_scale)

    def forward(
        self,
        grouped_input: Tensor,
        *,
        route_mode: RouteMode,
        temperature: float,
        top_k_leaves: int | None,
    ) -> Tensor:
        summary = torch.einsum("tgc,grc->tgr", grouped_input, self.summary_weight)
        if self.summary_bias is not None:
            summary = summary + self.summary_bias

        routing = self.router(
            summary,
            route_mode=route_mode,
            temperature=temperature,
            top_k_leaves=top_k_leaves,
        )
        stage_output = self.experts(grouped_input, routing)
        return self.stage_scale * stage_output


class ScaleTreeFFN(nn.Module):
    """Reusable tree-structured replacement for Transformer FFN/MLP layers."""

    def __init__(self, config: ScaleTreeFFNConfig) -> None:
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            [ScaleTreeStage(config) for _ in range(config.num_stages)]
        )
        self.output_dropout = nn.Dropout(config.output_dropout)

    def forward(
        self,
        x: Tensor,
        *,
        route_mode: RouteMode | None = None,
        temperature: float | None = None,
        top_k_leaves: int | None = None,
    ) -> Tensor:
        self._validate_input(x)
        resolved_mode = self._resolve_route_mode(route_mode)
        resolved_temperature = (
            self.config.router_temperature if temperature is None else temperature
        )
        resolved_top_k = (
            self.config.top_k_leaves if top_k_leaves is None else top_k_leaves
        )
        if resolved_top_k is not None and resolved_top_k > self.config.num_leaves:
            raise ValueError(
                "top_k_leaves cannot exceed the total number of leaves. "
                f"Got top_k_leaves={resolved_top_k}, num_leaves={self.config.num_leaves}."
            )

        flat_input = x.reshape(-1, self.config.embed_dim)
        grouped_input = flat_input.reshape(
            -1,
            self.config.num_groups,
            self.config.group_dim,
        )

        grouped_output = torch.zeros_like(grouped_input)
        for stage in self.stages:
            grouped_output = grouped_output + stage(
                grouped_input,
                route_mode=resolved_mode,
                temperature=resolved_temperature,
                top_k_leaves=resolved_top_k,
            )

        output = grouped_output.reshape(*x.shape[:-1], self.config.embed_dim)
        return self.output_dropout(output)

    def _resolve_route_mode(self, route_mode: RouteMode | None) -> RouteMode:
        if route_mode is not None:
            return route_mode
        if self.training:
            return self.config.training_route_mode
        return self.config.inference_route_mode

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim < 2:
            raise ValueError(
                "ScaleTreeFFN expects an input with at least 2 dimensions "
                f"(..., embed_dim). Got shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.config.embed_dim:
            raise ValueError(
                "Input embedding dimension does not match config. "
                f"Expected {self.config.embed_dim}, got {x.shape[-1]}."
            )

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.config.embed_dim}, "
            f"groups={self.config.num_groups}, "
            f"stages={self.config.num_stages}, "
            f"trees_per_group={self.config.trees_per_group}, "
            f"tree_width={self.config.num_leaves}, "
            f"tree_depth={self.config.effective_tree_depth}"
        )


class ScaleTreeTransformerBlock(nn.Module):
    """Minimal pre-norm Transformer block wrapper using ``ScaleTreeFFN``.

    The attention module is injected from outside so the block can be reused in
    ViT/DeiT codebases without forcing a particular attention implementation.
    The attention module is expected to accept a single tensor input and return
    a tensor with the same trailing embedding dimension.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        attention: nn.Module,
        tree_ffn: ScaleTreeFFN,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        residual_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("residual_dropout must be in [0, 1).")
        if tree_ffn.config.embed_dim != embed_dim:
            raise ValueError(
                "tree_ffn embed_dim must match block embed_dim. "
                f"Got tree_ffn={tree_ffn.config.embed_dim}, embed_dim={embed_dim}."
            )

        self.embed_dim = embed_dim
        self.attention = attention
        self.tree_ffn = tree_ffn
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self,
        x: Tensor,
        *,
        attn_kwargs: dict[str, Any] | None = None,
        route_mode: RouteMode | None = None,
        temperature: float | None = None,
        top_k_leaves: int | None = None,
    ) -> Tensor:
        self._validate_input(x)
        attention_args = {} if attn_kwargs is None else attn_kwargs

        attn_output = self.attention(self.norm1(x), **attention_args)
        if not isinstance(attn_output, torch.Tensor):
            raise TypeError(
                "attention module must return a torch.Tensor. "
                f"Got {type(attn_output)!r}."
            )
        x = x + self.residual_dropout(attn_output)

        ffn_output = self.tree_ffn(
            self.norm2(x),
            route_mode=route_mode,
            temperature=temperature,
            top_k_leaves=top_k_leaves,
        )
        x = x + self.residual_dropout(ffn_output)
        return x

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim < 2:
            raise ValueError(
                "ScaleTreeTransformerBlock expects an input with at least 2 "
                f"dimensions (..., embed_dim). Got shape={tuple(x.shape)}."
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                "Input embedding dimension does not match block embed_dim. "
                f"Expected {self.embed_dim}, got {x.shape[-1]}."
            )

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}"
