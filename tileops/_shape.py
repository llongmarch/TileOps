"""Shape helpers shared by facades.

The row-wise kernels in ``norm`` and ``reduction`` operate on a logical
``(M, N)`` view where the reduced axis is last.  These helpers centralize the
axis normalization / permutation code so facade modules do not import each
other's private functions.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def invert_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for out_dim, in_dim in enumerate(perm):
        inv[in_dim] = out_dim
    return tuple(inv)


def merge_on_dim(
    x: torch.Tensor,
    dim: int,
    *,
    op_name: str,
) -> tuple[torch.Tensor, int, int, tuple[int, ...], tuple[int, ...]]:
    """Permute *dim* last, return ``(M, N)`` view and metadata."""
    nd = x.ndim
    dim = dim if dim >= 0 else dim + nd
    if not 0 <= dim < nd:
        raise ValueError(f"{op_name}: invalid dim {dim} for ndim={nd}")
    n = x.shape[dim]
    if n <= 0:
        raise ValueError(f"{op_name}: reduced axis must have positive extent")
    perm = tuple(i for i in range(nd) if i != dim) + (dim,)
    x_perm = x.permute(*perm).contiguous()
    m = x_perm.numel() // n
    return x_perm.view(m, n), m, n, perm, tuple(x.shape)


def unmerge(
    y_2d: torch.Tensor,
    orig_shape: tuple[int, ...],
    perm: tuple[int, ...],
) -> torch.Tensor:
    sizes_perm = tuple(orig_shape[p] for p in perm)
    y_perm = y_2d.view(sizes_perm)
    inv = invert_permutation(perm)
    return y_perm.permute(*inv).contiguous()


def trailing_normalized(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
) -> tuple[torch.Tensor, int, int, tuple[int, ...]]:
    """Return ``x`` as ``(M, N)`` where ``N = prod(normalized_shape)``."""
    if not normalized_shape:
        raise ValueError("normalized_shape must be non-empty")
    if any(d <= 0 for d in normalized_shape):
        raise ValueError("normalized_shape entries must be positive")
    n = math.prod(normalized_shape)
    tail = x.shape[-len(normalized_shape):]
    if tuple(tail) != tuple(normalized_shape):
        raise ValueError(
            f"tensor shape {tuple(x.shape)} does not end with "
            f"normalized_shape {tuple(normalized_shape)}"
        )
    if x.numel() % n != 0:
        raise ValueError("internal shape error: numel not divisible by norm width")
    m = x.numel() // n
    return x.contiguous().view(m, n), m, n, tuple(x.shape)
