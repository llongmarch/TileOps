"""Sorting operators: ``sort``, ``argsort``.

Backed by TileLang kernels for per-row sort/argsort over the last dimension.
For arbitrary *dim*, the facade permutes the target axis last, invokes the
row-wise kernel, then permutes back.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.runtime import (
    BACKEND_PACKAGES,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_row_sort_kernel,
    target_kind,
    torch_to_tl_dtype,
)

import importlib


def _load_backend_modules(target: str):
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No sort backend for target kind {kind!r} ({target!r})"
        )
    try:
        return importlib.import_module(f"{pkg}.sort")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.sort for target {target!r}"
        ) from exc


def _validate(x: torch.Tensor, dim: int, op: str):
    if not x.dtype.is_floating_point:
        raise TypeError(f"{op}: input must have a floating dtype, got {x.dtype}")
    ndim = x.ndim
    dim_n = dim if dim >= 0 else dim + ndim
    if not 0 <= dim_n < ndim:
        raise ValueError(f"{op}: invalid dim {dim} for ndim={ndim}")
    return dim_n


def sort(
    input: torch.Tensor,
    dim: int = -1,
    descending: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sort elements along *dim* in ascending/descending order (``torch.sort``).

    Returns sorted values (same shape as *input*).
    """
    dim_n = _validate(input, dim, "sort")
    ndim = input.ndim
    if ndim <= 1:
        return _sort_1d(input, input.shape[-1], descending, out=out)

    # Permute sort dim to last.
    perm = tuple(i for i in range(ndim) if i != dim_n) + (dim_n,)
    x_perm = input.permute(*perm).contiguous()
    m = x_perm.numel() // x_perm.shape[-1]
    n = x_perm.shape[-1]

    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    mod = _load_backend_modules(tgt)
    kernel = mod.sort_row(
        m, n, descending,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )

    out_flat = torch.empty(m * n, dtype=input.dtype, device=input.device)
    invoke_row_sort_kernel(
        kernel, x_perm, out_flat,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result_perm = out_flat.reshape(m, n)
    # Permute back.
    inv = [0] * ndim
    for out_d, in_d in enumerate(perm):
        inv[in_d] = out_d
    result = result_perm.permute(*inv).contiguous()
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(
                f"sort: out.shape={tuple(out.shape)} != {tuple(input.shape)}"
            )
        out.copy_(result)
        return out
    return result


def argsort(
    input: torch.Tensor,
    dim: int = -1,
    descending: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return indices that sort *input* along *dim* (``torch.argsort``).

    Returns ``torch.int64`` indices (same shape as *input*).
    """
    dim_n = _validate(input, dim, "argsort")
    ndim = input.ndim
    if ndim <= 1:
        return _argsort_1d(input, input.shape[-1], descending, out=out)

    perm = tuple(i for i in range(ndim) if i != dim_n) + (dim_n,)
    x_perm = input.permute(*perm).contiguous()
    m = x_perm.numel() // x_perm.shape[-1]
    n = x_perm.shape[-1]

    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    mod = _load_backend_modules(tgt)
    kernel = mod.argsort_row(
        m, n, descending,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )

    idx_flat = torch.empty(m * n, dtype=torch.int64, device=input.device)
    invoke_row_sort_kernel(
        kernel, x_perm, idx_flat,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    result_perm = idx_flat.reshape(m, n)
    inv = [0] * ndim
    for out_d, in_d in enumerate(perm):
        inv[in_d] = out_d
    result = result_perm.permute(*inv).contiguous()
    if out is not None:
        if out.shape != input.shape:
            raise ValueError(
                f"argsort: out.shape={tuple(out.shape)} != {tuple(input.shape)}"
            )
        out.copy_(result)
        return out
    return result


def _sort_1d(
    x: torch.Tensor,
    n: int,
    descending: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    mod = _load_backend_modules(tgt)
    kernel = mod.sort_row(
        1, n, descending,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )
    out_flat = torch.empty(n, dtype=x.dtype, device=x.device)
    invoke_row_sort_kernel(
        kernel, x.view(1, n), out_flat,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    if out is not None:
        out.copy_(out_flat)
        return out
    return out_flat


def _argsort_1d(
    x: torch.Tensor,
    n: int,
    descending: bool,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    mod = _load_backend_modules(tgt)
    kernel = mod.argsort_row(
        1, n, descending,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )
    idx_flat = torch.empty(n, dtype=torch.int64, device=x.device)
    invoke_row_sort_kernel(
        kernel, x.view(1, n), idx_flat,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    if out is not None:
        out.copy_(idx_flat)
        return out
    return idx_flat


__all__ = ["sort", "argsort"]
