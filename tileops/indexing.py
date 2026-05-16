"""Indexing operators: ``gather``, ``index_select``, ``nonzero``.

``gather`` is backed by a TileLang kernel (row-indexed gather along last dim).
``index_select`` and ``nonzero`` delegate to PyTorch (view-only, no custom
kernel needed).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tileops.runtime import (
    BACKEND_PACKAGES,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_gather_kernel,
    target_kind,
    torch_to_tl_dtype,
)

import importlib


def _load_backend_fn(fn_name: str, target: str):
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No indexing backend for target kind {kind!r} ({target!r})"
        )
    try:
        mod = importlib.import_module(f"{pkg}.indexing")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.indexing for target {target!r}"
        ) from exc
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.indexing has no {fn_name}() for {target!r}"
        )
    return fn


def gather(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather elements along *dim* specified by *index* (``torch.gather``).

    Shapes: *input* and *index* have the same ndim; for each dimension *d*
    where ``d != dim``, ``index.shape[d] <= input.shape[d]``.
    The output has the same shape as *index*.
    """
    if index.dtype != torch.int64:
        raise TypeError("gather: index must be torch.int64")
    if input.ndim != index.ndim:
        raise ValueError(
            f"gather: input.ndim={input.ndim} != index.ndim={index.ndim}"
        )
    for d in range(input.ndim):
        if d != dim and index.shape[d] > input.shape[d]:
            raise ValueError(
                f"gather: index.shape[{d}]={index.shape[d]} > "
                f"input.shape[{d}]={input.shape[d]}"
            )
    if input.device != index.device:
        raise TypeError("gather: input and index must share device")

    # Normalise dim.
    ndim = input.ndim
    dim_n = dim if dim >= 0 else dim + ndim
    if not 0 <= dim_n < ndim:
        raise ValueError(f"gather: invalid dim {dim} for ndim={ndim}")

    perm = tuple(i for i in range(ndim) if i != dim_n) + (dim_n,)
    input_perm = input.permute(*perm).contiguous()
    index_perm = index.permute(*perm).contiguous()

    m = input_perm.numel() // input_perm.shape[-1]
    n_data = input_perm.shape[-1]
    n_index = index_perm.shape[-1]

    if out is None:
        out = torch.empty_like(index, dtype=input.dtype, device=input.device)
    out_perm = out.permute(*perm).contiguous()

    tl_dtype = torch_to_tl_dtype(input.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)

    backend_fn = _load_backend_fn("gather", tgt)
    kernel = backend_fn(
        m, n_data, n_index,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )
    invoke_gather_kernel(
        kernel, input_perm, index_perm, out=out_perm,
        tilelang_target=tgt,
        execution_backend=eb,
    )
    # Permute back.
    inv_perm = tuple(i for i in range(ndim) if i != dim_n) + (dim_n,)
    inv = [0] * ndim
    for out_dim, in_dim in enumerate(inv_perm):
        inv[in_dim] = out_dim
    result = out_perm.permute(*inv).contiguous()
    if out.data_ptr() != result.data_ptr():
        out.copy_(result)
    return out


def index_select(
    input: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select elements along *dim* by *index* (``torch.index_select``).

    Delegates to PyTorch (no custom kernel).
    """
    if out is not None:
        return torch.index_select(input, dim, index, out=out)
    return torch.index_select(input, dim, index)


def nonzero(
    input: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    as_tuple: bool = False,
):
    """Return indices of non-zero elements (``torch.nonzero``).

    Delegates to PyTorch (no custom kernel).
    """
    if as_tuple:
        return torch.nonzero(input, as_tuple=True)
    if out is not None:
        return torch.nonzero(input, out=out)
    return torch.nonzero(input)


__all__ = ["gather", "index_select", "nonzero"]
