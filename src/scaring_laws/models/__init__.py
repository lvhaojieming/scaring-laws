"""Reusable model components for ScaleTree-style architectures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["ScaleTreeFFN", "ScaleTreeFFNConfig", "ScaleTreeTransformerBlock"]

if TYPE_CHECKING:
    from .scaletree import ScaleTreeFFN, ScaleTreeFFNConfig, ScaleTreeTransformerBlock


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .scaletree import ScaleTreeFFN, ScaleTreeFFNConfig, ScaleTreeTransformerBlock

    exports = {
        "ScaleTreeFFN": ScaleTreeFFN,
        "ScaleTreeFFNConfig": ScaleTreeFFNConfig,
        "ScaleTreeTransformerBlock": ScaleTreeTransformerBlock,
    }
    return exports[name]
