"""Top-*k* along a dimension (``topk``).

:class:`topk` mirrors ``torch.topk`` and uses the same *merge dim to last
axis* layout as :mod:`tileops.reduction`.  The per-row insertion-sort kernel
lives in the backend modules (``tileops.cuda.reduce`` / ``tileops.metal.reduce``).

Supported dtypes
----------------
* ``float32``, ``float16``, ``bfloat16``.
* Indices are ``torch.int64`` (like PyTorch).

Tie-breaking follows :func:`tileops.reduction.reduce_argmax` semantics: the
first occurrence of a value wins.
"""

from __future__ import annotations

import torch

from tileops._infra import dispatch_compile_reduce
from tileops._shape import merge_on_dim, unmerge
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_row_topk_kernel,
    torch_to_tl_dtype,
)

_REDUCE_THREADS = 1


def topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-*k* along *dim* (``torch.topk``-compatible).

    Returns ``(values, indices)`` with the same rank as *input* and size *k*
    along *dim*.  Tie-breaking matches ``reduce_argmax``: the first index wins.
    The TileLang kernel always returns elements in descending (``largest=True``)
    or ascending (``largest=False``) order; when *sorted* is ``False`` the order
    is still sorted (stricter than PyTorch for ``sorted=False``).
    """
    if not input.dtype.is_floating_point:
        raise TypeError(
            f"topk: input must have a floating dtype (got {input.dtype})"
        )
    if k <= 0:
        raise ValueError(f"topk: k must be positive (got {k})")
    nd = input.ndim
    dim_n = dim if dim >= 0 else dim + nd
    if not 0 <= dim_n < nd:
        raise ValueError(f"topk: invalid dim {dim} for ndim={nd}")
    n = input.shape[dim_n]
    if k > n:
        raise ValueError(
            f"topk: k ({k}) cannot exceed dimension size ({n})"
        )

    x_2d, m, cols, perm, orig = merge_on_dim(input, dim_n, op_name="topk")
    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    kernel = dispatch_compile_reduce(
        op_name="topk",
        target=tgt,
        rows=m,
        cols=cols,
        threads=_REDUCE_THREADS,
        dtype=tl_dtype,
        execution_backend=eb,
        k=k,
        largest=largest,
    )
    out_shape = _topk_output_shape(orig, dim_n, k)
    vals_1d = torch.empty(m * k, dtype=input.dtype, device=input.device)
    idx_1d = torch.empty(m * k, dtype=torch.int64, device=input.device)
    invoke_row_topk_kernel(
        kernel, x_2d, vals_1d, idx_1d,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    vals = unmerge(vals_1d.view(m, k), out_shape, perm)
    indices = unmerge(idx_1d.view(m, k), out_shape, perm)
    return vals, indices


def _topk_output_shape(
    orig_shape: tuple[int, ...],
    dim: int,
    k: int,
) -> tuple[int, ...]:
    shape = list(orig_shape)
    shape[dim] = k
    return tuple(shape)


__all__ = [
    "topk",
]
