"""Conditional operators: ``where``, ``masked_fill``.

``where(condition, x, y)`` and ``masked_fill(x, mask, value)`` are backed by
TileLang kernels.  ``condition`` / ``mask`` can be any non-zero-means-true tensor
(boolean or float); they are converted to ``float32`` internally.
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
    suggest_tile_config,
    target_kind,
    torch_to_tl_dtype,
)

import importlib


def _load_backend_module(target: str):
    kind = target_kind(target)
    pkg = BACKEND_PACKAGES.get(kind)
    if pkg is None:
        raise NotImplementedError(
            f"No condition backend for target kind {kind!r} ({target!r})"
        )
    try:
        return importlib.import_module(f"{pkg}.condition")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import {pkg}.condition for target {target!r}"
        ) from exc


def _normalise_cond(c: torch.Tensor) -> torch.Tensor:
    """Convert condition/mask to contiguous float32 1-D."""
    return c.contiguous().float().view(-1)


def where(
    condition: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Element-wise ``condition ? x : y`` (``torch.where``).

    All three inputs must share shape and device.
    """
    if condition.shape != x.shape or x.shape != y.shape:
        raise ValueError(
            f"where: condition.shape={tuple(condition.shape)} != "
            f"x.shape={tuple(x.shape)} or != y.shape={tuple(y.shape)}"
        )
    if x.dtype != y.dtype:
        raise TypeError("where: x and y dtypes must match")
    if condition.device != x.device or x.device != y.device:
        raise TypeError("where: all inputs must share device")
    if x.numel() == 0:
        raise ValueError("where: input tensors are empty")

    shape = x.shape
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    bs, threads = suggest_tile_config(x.numel())

    mod = _load_backend_module(tgt)
    kernel = mod.where(
        shape, bs, threads,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
    )

    cond_f = _normalise_cond(condition)
    xf = x.contiguous().view(-1)
    yf = y.contiguous().view(-1)
    if out is None:
        out = torch.empty_like(x)
    of = out.contiguous().view(-1)

    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(cond_f, xf, yf, of)
    else:
        ret = kernel(cond_f, xf, yf)
        if ret is not None:
            of.copy_(ret)
        else:
            kernel(cond_f, xf, yf, of)
    if of.data_ptr() != out.data_ptr():
        out.copy_(of)
    return out


def masked_fill(
    x: torch.Tensor,
    mask: torch.Tensor,
    value: float,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fill elements of *x* with *value* wherever *mask* is non-zero.

    Shapes: *mask* must be broadcastable to *x* shape.
    """
    if mask.shape != x.shape:
        # Try broadcasting.
        try:
            torch.broadcast_shapes(mask.shape, x.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"masked_fill: mask.shape={tuple(mask.shape)} not "
                f"broadcastable to x.shape={tuple(x.shape)}"
            ) from exc
        # Broadcast at PyTorch level.
        mask_b = torch.broadcast_to(mask, x.shape)
        if out is None:
            result = x.clone()
        else:
            if out.shape != x.shape:
                raise ValueError(
                    f"masked_fill: out.shape={tuple(out.shape)} != "
                    f"x.shape={tuple(x.shape)}"
                )
            result = out
        result.masked_fill_(mask_b, value)
        return result

    if x.numel() == 0:
        raise ValueError("masked_fill: input tensor is empty")
    if x.device != mask.device:
        raise TypeError("masked_fill: x and mask must share device")

    shape = x.shape
    tl_dtype = torch_to_tl_dtype(x.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    bs, threads = suggest_tile_config(x.numel())

    mod = _load_backend_module(tgt)
    kernel = mod.masked_fill(
        shape, bs, threads,
        dtype=tl_dtype,
        target=tgt,
        execution_backend=eb,
        value=float(value),
    )

    mask_f = _normalise_cond(mask)
    xf = x.contiguous().view(-1)
    if out is None:
        out = torch.empty_like(x)
    of = out.contiguous().view(-1)

    kind = target_kind(tgt)
    if kind == "metal" and eb == "torch":
        kernel(xf, mask_f, of)
    else:
        ret = kernel(xf, mask_f)
        if ret is not None:
            of.copy_(ret)
        else:
            kernel(xf, mask_f, of)
    if of.data_ptr() != out.data_ptr():
        out.copy_(of)
    return out


__all__ = ["where", "masked_fill"]
