"""Matrix triangular operators: ``tril``, ``triu``.

Backed by TileLang kernels that operate on 2-D inputs.
For batched inputs (``ndim > 2``), the last two dims are treated as
the matrix and the leading dims are iterated in PyTorch.
"""

from __future__ import annotations

from typing import Optional

import torch

from tileops.runtime import (
    BACKEND_PACKAGES,
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_tril_triu_kernel,
    target_kind,
    torch_to_tl_dtype,
)

import importlib


def _load_backend_fn(fn_name: str, target: str):
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No matrix backend for target kind {kind!r} ({target!r})"
        )
    try:
        mod = importlib.import_module(f"{pkg}.matrix")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.matrix for target {target!r}"
        ) from exc
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise NotImplementedError(
            f"{pkg}.matrix has no {fn_name}() for {target!r}"
        )
    return fn


def tril(
    input: torch.Tensor,
    diagonal: int = 0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lower triangular part of a matrix (``torch.tril``).

    For ``ndim > 2``, applies to the last two dimensions.
    """
    if input.ndim < 2:
        raise ValueError(
            f"tril: input must have at least 2 dims, got {input.ndim}"
        )
    if input.ndim == 2:
        return _tril_2d(input, diagonal, out=out)

    # Batched: loop over leading dims, apply 2-D kernel per slice.
    *batch, m, n = input.shape
    flat = input.reshape(-1, m, n)
    outs = []
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    for i in range(flat.shape[0]):
        slice_out = _tril_2d(flat[i], diagonal, out=None,
                              _tgt=tgt, _eb=eb)
        outs.append(slice_out)
    result = torch.stack(outs, dim=0).reshape(*batch, m, n)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"tril: out.shape={tuple(out.shape)} != {tuple(result.shape)}"
            )
        out.copy_(result)
        return out
    return result


def triu(
    input: torch.Tensor,
    diagonal: int = 0,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Upper triangular part of a matrix (``torch.triu``).

    For ``ndim > 2``, applies to the last two dimensions.
    """
    if input.ndim < 2:
        raise ValueError(
            f"triu: input must have at least 2 dims, got {input.ndim}"
        )
    if input.ndim == 2:
        return _triu_2d(input, diagonal, out=out)

    *batch, m, n = input.shape
    flat = input.reshape(-1, m, n)
    outs = []
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    for i in range(flat.shape[0]):
        slice_out = _triu_2d(flat[i], diagonal, out=None,
                              _tgt=tgt, _eb=eb)
        outs.append(slice_out)
    result = torch.stack(outs, dim=0).reshape(*batch, m, n)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"triu: out.shape={tuple(out.shape)} != {tuple(result.shape)}"
            )
        out.copy_(result)
        return out
    return result


def _tril_2d(
    x: torch.Tensor,
    diagonal: int = 0,
    out: Optional[torch.Tensor] = None,
    _tgt: Optional[str] = None,
    _eb: Optional[str] = None,
) -> torch.Tensor:
    m, n = x.shape
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = _tgt if _tgt is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = _eb if _eb is not None else default_execution_backend(tgt, dev)
    backend_fn = _load_backend_fn("tril", tgt)
    kernel = backend_fn(
        m, n, diagonal,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )
    return invoke_tril_triu_kernel(
        kernel, x, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
    )


def _triu_2d(
    x: torch.Tensor,
    diagonal: int = 0,
    out: Optional[torch.Tensor] = None,
    _tgt: Optional[str] = None,
    _eb: Optional[str] = None,
) -> torch.Tensor:
    m, n = x.shape
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = _tgt if _tgt is not None else default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = _eb if _eb is not None else default_execution_backend(tgt, dev)
    backend_fn = _load_backend_fn("triu", tgt)
    kernel = backend_fn(
        m, n, diagonal,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )
    return invoke_tril_triu_kernel(
        kernel, x, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
    )


__all__ = ["tril", "triu"]
