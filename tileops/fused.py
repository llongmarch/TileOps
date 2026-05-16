"""Fused LLM helpers: activation×gate **and** residual + norm pairs.

Single-kernel gated activations::

* :func:`silu_and_mul` — ``silu(a) * b``
* :func:`gelu_and_mul` — ``gelu(a, tanh approx) * b``

Residual-connected normalisation (delegates to :mod:`tileops.norm`; same
kernels as ``skip_*`` in the norm backends)::

* :func:`skip_rms_norm` — ``rms_norm(x + residual, …)``
* :func:`skip_layer_norm` — ``layer_norm(x + residual, …)``
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from tileops.runtime import dispatch_compile
from tileops.norm import skip_layer_norm, skip_rms_norm
from tileops.runtime import (
    default_execution_backend,
    default_tilelang_target,
    default_torch_device,
    invoke_kernel,
    suggest_tile_config,
    torch_to_tl_dtype,
)


def _validate_fused(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor],
    op: str,
) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{op}: shape mismatch — a.shape={a.shape}, b.shape={b.shape}"
        )
    if a.dtype != b.dtype:
        raise TypeError(
            f"{op}: dtype mismatch — a.dtype={a.dtype}, b.dtype={b.dtype}"
        )
    if a.device != b.device:
        raise ValueError(
            f"{op}: device mismatch — a.device={a.device}, b.device={b.device}"
        )
    if out is not None:
        if out.shape != a.shape:
            raise ValueError(
                f"{op}: out.shape={out.shape} != input shape {a.shape}"
            )
        if out.dtype != a.dtype:
            raise TypeError(
                f"{op}: out.dtype={out.dtype} != input dtype {a.dtype}"
            )
        if out.device != a.device:
            raise ValueError(
                f"{op}: out.device={out.device} != input device {a.device}"
            )


def _dispatch_fused(
    op_name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_fused(a, b, out, op_name)
    tl_dtype = torch_to_tl_dtype(a.dtype)
    tgt = default_tilelang_target()
    dev = default_torch_device(tgt)
    eb = default_execution_backend(tgt, dev)
    shape = a.shape
    bs, threads = suggest_tile_config(math.prod(shape))
    kernel = dispatch_compile(
        module_name="fused",
        op_name=op_name,
        target=tgt,
        shape=shape,
        block_size=bs,
        threads=threads,
        dtype=tl_dtype,
        execution_backend=eb,
    )
    return invoke_kernel(
        kernel, a, b, out=out,
        tilelang_target=tgt,
        execution_backend=eb,
    )


def silu_and_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``silu(a) * b`` element-wise (same shapes)."""
    return _dispatch_fused("silu_and_mul", a, b, out=out)


def gelu_and_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``gelu(a) * b`` with tanh GELU approximation (matches :func:`tileops.gelu`)."""
    return _dispatch_fused("gelu_and_mul", a, b, out=out)


__all__ = [
    "gelu_and_mul",
    "silu_and_mul",
    "skip_layer_norm",
    "skip_rms_norm",
]
