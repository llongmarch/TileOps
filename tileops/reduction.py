"""Reductions along a single dimension (``reduce_sum``, ``reduce_mean``, …).

Uses the same *merge dim to last axis* layout as :mod:`tileops.norm`.  Names use
the ``reduce_*`` prefix so ``from tileops import reduce_sum`` does not shadow
Python's built-in :func:`sum`.

Supported dtypes match the rest of TileOps: ``float32``, ``float16``,
``bfloat16``.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops._infra import dispatch_compile_reduce
from tileops._shape import merge_on_dim
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_row_reduce_kernel,
    torch_to_tl_dtype,
)

_REDUCE_THREADS = 1

_PUBLIC_NAMES: dict[str, str] = {
    "sum": "reduce_sum",
    "mean": "reduce_mean",
    "prod": "reduce_prod",
    "amax": "reduce_amax",
    "amin": "reduce_amin",
}


def _reduce_output_shape(
    orig_shape: tuple[int, ...],
    dim: int,
    *,
    keepdim: bool,
) -> tuple[int, ...]:
    if keepdim:
        return orig_shape[:dim] + (1,) + orig_shape[dim + 1:]
    return orig_shape[:dim] + orig_shape[dim + 1:]


def _reduce_dispatch(
    op_name: str,
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    dim_n = dim if dim >= 0 else dim + x.ndim
    pub = _PUBLIC_NAMES[op_name]
    x_2d, m, n, _perm, orig = merge_on_dim(x, dim, op_name=pub)
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name=op_name,
        target=tgt,
        rows=m,
        cols=n,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    out_shape = _reduce_output_shape(orig, dim_n, keepdim=keepdim)
    out_1d = torch.empty(m, dtype=x.dtype, device=x.device)
    invoke_row_reduce_kernel(
        kernel, x_2d, out_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result = out_1d.reshape(out_shape)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"{pub}: out.shape={tuple(out.shape)} != "
                f"{tuple(result.shape)}"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise TypeError(f"{pub}: out dtype/device must match input")
        out.copy_(result)
        return out
    return result


def reduce_sum(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of *x* along *dim* (like ``torch.sum(x, dim=…)``)."""
    return _reduce_dispatch("sum", x, dim, keepdim=keepdim, out=out)


def reduce_mean(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean of *x* along *dim* (like ``torch.mean(x, dim=…)``)."""
    return _reduce_dispatch("mean", x, dim, keepdim=keepdim, out=out)


def reduce_prod(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Product of *x* along *dim* (like ``torch.prod(x, dim=…)``)."""
    return _reduce_dispatch("prod", x, dim, keepdim=keepdim, out=out)


def reduce_amax(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Maximum of *x* along *dim* (like ``torch.amax(x, dim=…)``)."""
    return _reduce_dispatch("amax", x, dim, keepdim=keepdim, out=out)


def reduce_amin(
    x: torch.Tensor,
    dim: int,
    *,
    keepdim: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Minimum of *x* along *dim* (like ``torch.amin(x, dim=…)``)."""
    return _reduce_dispatch("amin", x, dim, keepdim=keepdim, out=out)


__all__ = [
    "reduce_sum",
    "reduce_mean",
    "reduce_prod",
    "reduce_amax",
    "reduce_amin",
]
